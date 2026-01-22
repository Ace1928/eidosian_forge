import re
from collections import defaultdict
from . import Tree, Token
from .common import ParserConf
from .parsers import earley
from .grammar import Rule, Terminal, NonTerminal
class TreeMatcher:
    """Match the elements of a tree node, based on an ontology
    provided by a Lark grammar.

    Supports templates and inlined rules (`rule{a, b,..}` and `_rule`)

    Initialize with an instance of Lark.
    """

    def __init__(self, parser):
        assert not parser.options.maybe_placeholders
        self.tokens, rules, _extra = parser.grammar.compile(parser.options.start, set())
        self.rules_for_root = defaultdict(list)
        self.rules = list(self._build_recons_rules(rules))
        self.rules.reverse()
        self.rules = _best_rules_from_group(self.rules)
        self.parser = parser
        self._parser_cache = {}

    def _build_recons_rules(self, rules):
        """Convert tree-parsing/construction rules to tree-matching rules"""
        expand1s = {r.origin for r in rules if r.options.expand1}
        aliases = defaultdict(list)
        for r in rules:
            if r.alias:
                aliases[r.origin].append(r.alias)
        rule_names = {r.origin for r in rules}
        nonterminals = {sym for sym in rule_names if sym.name.startswith('_') or sym in expand1s or sym in aliases}
        seen = set()
        for r in rules:
            recons_exp = [sym if sym in nonterminals else Terminal(sym.name) for sym in r.expansion if not is_discarded_terminal(sym)]
            if recons_exp == [r.origin] and r.alias is None:
                continue
            sym = NonTerminal(r.alias) if r.alias else r.origin
            rule = make_recons_rule(sym, recons_exp, r.expansion)
            if sym in expand1s and len(recons_exp) != 1:
                self.rules_for_root[sym.name].append(rule)
                if sym.name not in seen:
                    yield make_recons_rule_to_term(sym, sym)
                    seen.add(sym.name)
            elif sym.name.startswith('_') or sym in expand1s:
                yield rule
            else:
                self.rules_for_root[sym.name].append(rule)
        for origin, rule_aliases in aliases.items():
            for alias in rule_aliases:
                yield make_recons_rule_to_term(origin, NonTerminal(alias))
            yield make_recons_rule_to_term(origin, origin)

    def match_tree(self, tree, rulename):
        """Match the elements of `tree` to the symbols of rule `rulename`.

        Parameters:
            tree (Tree): the tree node to match
            rulename (str): The expected full rule name (including template args)

        Returns:
            Tree: an unreduced tree that matches `rulename`

        Raises:
            UnexpectedToken: If no match was found.

        Note:
            It's the callers' responsibility match the tree recursively.
        """
        if rulename:
            name, _args = parse_rulename(rulename)
            assert tree.data == name
        else:
            rulename = tree.data
        try:
            parser = self._parser_cache[rulename]
        except KeyError:
            rules = self.rules + _best_rules_from_group(self.rules_for_root[rulename])
            callbacks = {rule: rule.alias for rule in rules}
            conf = ParserConf(rules, callbacks, [rulename])
            parser = earley.Parser(self.parser.lexer_conf, conf, _match, resolve_ambiguity=True)
            self._parser_cache[rulename] = parser
        unreduced_tree = parser.parse(ChildrenLexer(tree.children), rulename)
        assert unreduced_tree.data == rulename
        return unreduced_tree