import hashlib
import os.path
import sys
from collections import namedtuple
from copy import copy, deepcopy
import pkgutil
from ast import literal_eval
from contextlib import suppress
from typing import List, Tuple, Union, Callable, Dict, Optional, Sequence, Generator
from .utils import bfs, logger, classify_bool, is_id_continue, is_id_start, bfs_all_unique, small_factors, OrderedSet
from .lexer import Token, TerminalDef, PatternStr, PatternRE, Pattern
from .parse_tree_builder import ParseTreeBuilder
from .parser_frontends import ParsingFrontend
from .common import LexerConf, ParserConf
from .grammar import RuleOptions, Rule, Terminal, NonTerminal, Symbol, TOKEN_DEFAULT_PRIORITY
from .utils import classify, dedup_list
from .exceptions import GrammarError, UnexpectedCharacters, UnexpectedToken, ParseError, UnexpectedInput
from .tree import Tree, SlottedTree as ST
from .visitors import Transformer, Visitor, v_args, Transformer_InPlace, Transformer_NonRecursive
class GrammarBuilder:
    global_keep_all_tokens: bool
    import_paths: List[Union[str, Callable]]
    used_files: Dict[str, str]
    _definitions: Dict[str, Definition]
    _ignore_names: List[str]

    def __init__(self, global_keep_all_tokens: bool=False, import_paths: Optional[List[Union[str, Callable]]]=None, used_files: Optional[Dict[str, str]]=None) -> None:
        self.global_keep_all_tokens = global_keep_all_tokens
        self.import_paths = import_paths or []
        self.used_files = used_files or {}
        self._definitions: Dict[str, Definition] = {}
        self._ignore_names: List[str] = []

    def _grammar_error(self, is_term, msg, *names):
        args = {}
        for i, name in enumerate(names, start=1):
            postfix = '' if i == 1 else str(i)
            args['name' + postfix] = name
            args['type' + postfix] = lowercase_type = ('rule', 'terminal')[is_term]
            args['Type' + postfix] = lowercase_type.title()
        raise GrammarError(msg.format(**args))

    def _check_options(self, is_term, options):
        if is_term:
            if options is None:
                options = 1
            elif not isinstance(options, int):
                raise GrammarError("Terminal require a single int as 'options' (e.g. priority), got %s" % (type(options),))
        else:
            if options is None:
                options = RuleOptions()
            elif not isinstance(options, RuleOptions):
                raise GrammarError("Rules require a RuleOptions instance as 'options'")
            if self.global_keep_all_tokens:
                options.keep_all_tokens = True
        return options

    def _define(self, name, is_term, exp, params=(), options=None, *, override=False):
        if name in self._definitions:
            if not override:
                self._grammar_error(is_term, "{Type} '{name}' defined more than once", name)
        elif override:
            self._grammar_error(is_term, 'Cannot override a nonexisting {type} {name}', name)
        if name.startswith('__'):
            self._grammar_error(is_term, 'Names starting with double-underscore are reserved (Error at {name})', name)
        self._definitions[name] = Definition(is_term, exp, params, self._check_options(is_term, options))

    def _extend(self, name, is_term, exp, params=(), options=None):
        if name not in self._definitions:
            self._grammar_error(is_term, "Can't extend {type} {name} as it wasn't defined before", name)
        d = self._definitions[name]
        if is_term != d.is_term:
            self._grammar_error(is_term, 'Cannot extend {type} {name} - one is a terminal, while the other is not.', name)
        if tuple(params) != d.params:
            self._grammar_error(is_term, 'Cannot extend {type} with different parameters: {name}', name)
        if d.tree is None:
            self._grammar_error(is_term, "Can't extend {type} {name} - it is abstract.", name)
        base = d.tree
        assert isinstance(base, Tree) and base.data == 'expansions'
        base.children.insert(0, exp)

    def _ignore(self, exp_or_name):
        if isinstance(exp_or_name, str):
            self._ignore_names.append(exp_or_name)
        else:
            assert isinstance(exp_or_name, Tree)
            t = exp_or_name
            if t.data == 'expansions' and len(t.children) == 1:
                t2, = t.children
                if t2.data == 'expansion' and len(t2.children) == 1:
                    item, = t2.children
                    if item.data == 'value':
                        item, = item.children
                        if isinstance(item, Terminal):
                            self._ignore_names.append(item.name)
                            return
            name = '__IGNORE_%d' % len(self._ignore_names)
            self._ignore_names.append(name)
            self._definitions[name] = Definition(True, t, options=TOKEN_DEFAULT_PRIORITY)

    def _unpack_import(self, stmt, grammar_name):
        if len(stmt.children) > 1:
            path_node, arg1 = stmt.children
        else:
            path_node, = stmt.children
            arg1 = None
        if isinstance(arg1, Tree):
            dotted_path = tuple(path_node.children)
            names = arg1.children
            aliases = dict(zip(names, names))
        else:
            dotted_path = tuple(path_node.children[:-1])
            if not dotted_path:
                name, = path_node.children
                raise GrammarError('Nothing was imported from grammar `%s`' % name)
            name = path_node.children[-1]
            aliases = {name.value: (arg1 or name).value}
        if path_node.data == 'import_lib':
            base_path = None
        else:
            if grammar_name == '<string>':
                try:
                    base_file = os.path.abspath(sys.modules['__main__'].__file__)
                except AttributeError:
                    base_file = None
            else:
                base_file = grammar_name
            if base_file:
                if isinstance(base_file, PackageResource):
                    base_path = PackageResource(base_file.pkg_name, os.path.split(base_file.path)[0])
                else:
                    base_path = os.path.split(base_file)[0]
            else:
                base_path = os.path.abspath(os.path.curdir)
        return (dotted_path, base_path, aliases)

    def _unpack_definition(self, tree, mangle):
        if tree.data == 'rule':
            name, params, exp, opts = _make_rule_tuple(*tree.children)
            is_term = False
        else:
            name = tree.children[0].value
            params = ()
            opts = int(tree.children[1]) if len(tree.children) == 3 else TOKEN_DEFAULT_PRIORITY
            exp = tree.children[-1]
            is_term = True
        if mangle is not None:
            params = tuple((mangle(p) for p in params))
            name = mangle(name)
        exp = _mangle_definition_tree(exp, mangle)
        return (name, is_term, exp, params, opts)

    def load_grammar(self, grammar_text: str, grammar_name: str='<?>', mangle: Optional[Callable[[str], str]]=None) -> None:
        tree = _parse_grammar(grammar_text, grammar_name)
        imports: Dict[Tuple[str, ...], Tuple[Optional[str], Dict[str, str]]] = {}
        for stmt in tree.children:
            if stmt.data == 'import':
                dotted_path, base_path, aliases = self._unpack_import(stmt, grammar_name)
                try:
                    import_base_path, import_aliases = imports[dotted_path]
                    assert base_path == import_base_path, 'Inconsistent base_path for %s.' % '.'.join(dotted_path)
                    import_aliases.update(aliases)
                except KeyError:
                    imports[dotted_path] = (base_path, aliases)
        for dotted_path, (base_path, aliases) in imports.items():
            self.do_import(dotted_path, base_path, aliases, mangle)
        for stmt in tree.children:
            if stmt.data in ('term', 'rule'):
                self._define(*self._unpack_definition(stmt, mangle))
            elif stmt.data == 'override':
                r, = stmt.children
                self._define(*self._unpack_definition(r, mangle), override=True)
            elif stmt.data == 'extend':
                r, = stmt.children
                self._extend(*self._unpack_definition(r, mangle))
            elif stmt.data == 'ignore':
                if mangle is None:
                    self._ignore(*stmt.children)
            elif stmt.data == 'declare':
                for symbol in stmt.children:
                    assert isinstance(symbol, Symbol), symbol
                    is_term = isinstance(symbol, Terminal)
                    if mangle is None:
                        name = symbol.name
                    else:
                        name = mangle(symbol.name)
                    self._define(name, is_term, None)
            elif stmt.data == 'import':
                pass
            else:
                assert False, stmt
        term_defs = {name: d.tree for name, d in self._definitions.items() if d.is_term}
        resolve_term_references(term_defs)

    def _remove_unused(self, used):

        def rule_dependencies(symbol):
            try:
                d = self._definitions[symbol]
            except KeyError:
                return []
            if d.is_term:
                return []
            return _find_used_symbols(d.tree) - set(d.params)
        _used = set(bfs(used, rule_dependencies))
        self._definitions = {k: v for k, v in self._definitions.items() if k in _used}

    def do_import(self, dotted_path: Tuple[str, ...], base_path: Optional[str], aliases: Dict[str, str], base_mangle: Optional[Callable[[str], str]]=None) -> None:
        assert dotted_path
        mangle = _get_mangle('__'.join(dotted_path), aliases, base_mangle)
        grammar_path = os.path.join(*dotted_path) + EXT
        to_try = self.import_paths + ([base_path] if base_path is not None else []) + [stdlib_loader]
        for source in to_try:
            try:
                if callable(source):
                    joined_path, text = source(base_path, grammar_path)
                else:
                    joined_path = os.path.join(source, grammar_path)
                    with open(joined_path, encoding='utf8') as f:
                        text = f.read()
            except IOError:
                continue
            else:
                h = sha256_digest(text)
                if self.used_files.get(joined_path, h) != h:
                    raise RuntimeError('Grammar file was changed during importing')
                self.used_files[joined_path] = h
                gb = GrammarBuilder(self.global_keep_all_tokens, self.import_paths, self.used_files)
                gb.load_grammar(text, joined_path, mangle)
                gb._remove_unused(map(mangle, aliases))
                for name in gb._definitions:
                    if name in self._definitions:
                        raise GrammarError("Cannot import '%s' from '%s': Symbol already defined." % (name, grammar_path))
                self._definitions.update(**gb._definitions)
                break
        else:
            open(grammar_path, encoding='utf8')
            assert False, "Couldn't import grammar %s, but a corresponding file was found at a place where lark doesn't search for it" % (dotted_path,)

    def validate(self) -> None:
        for name, d in self._definitions.items():
            params = d.params
            exp = d.tree
            for i, p in enumerate(params):
                if p in self._definitions:
                    raise GrammarError('Template Parameter conflicts with rule %s (in template %s)' % (p, name))
                if p in params[:i]:
                    raise GrammarError('Duplicate Template Parameter %s (in template %s)' % (p, name))
            if exp is None:
                continue
            for temp in exp.find_data('template_usage'):
                sym = temp.children[0].name
                args = temp.children[1:]
                if sym not in params:
                    if sym not in self._definitions:
                        self._grammar_error(d.is_term, "Template '%s' used but not defined (in {type} {name})" % sym, name)
                    if len(args) != len(self._definitions[sym].params):
                        expected, actual = (len(self._definitions[sym].params), len(args))
                        self._grammar_error(d.is_term, 'Wrong number of template arguments used for {name} (expected %s, got %s) (in {type2} {name2})' % (expected, actual), sym, name)
            for sym in _find_used_symbols(exp):
                if sym not in self._definitions and sym not in params:
                    self._grammar_error(d.is_term, "{Type} '{name}' used but not defined (in {type2} {name2})", sym, name)
        if not set(self._definitions).issuperset(self._ignore_names):
            raise GrammarError('Terminals %s were marked to ignore but were not defined!' % (set(self._ignore_names) - set(self._definitions)))

    def build(self) -> Grammar:
        self.validate()
        rule_defs = []
        term_defs = []
        for name, d in self._definitions.items():
            params, exp, options = (d.params, d.tree, d.options)
            if d.is_term:
                assert len(params) == 0
                term_defs.append((name, (exp, options)))
            else:
                rule_defs.append((name, params, exp, options))
        return Grammar(rule_defs, term_defs, self._ignore_names)