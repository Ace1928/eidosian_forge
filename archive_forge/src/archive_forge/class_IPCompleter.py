from __future__ import annotations
import builtins as builtin_mod
import enum
import glob
import inspect
import itertools
import keyword
import os
import re
import string
import sys
import tokenize
import time
import unicodedata
import uuid
import warnings
from ast import literal_eval
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property, partial
from types import SimpleNamespace
from typing import (
from IPython.core.guarded_eval import guarded_eval, EvaluationContext
from IPython.core.error import TryNext
from IPython.core.inputtransformer2 import ESC_MAGIC
from IPython.core.latex_symbols import latex_symbols, reverse_latex_symbol
from IPython.core.oinspect import InspectColors
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils import generics
from IPython.utils.decorators import sphinx_options
from IPython.utils.dir2 import dir2, get_real_method
from IPython.utils.docs import GENERATING_DOCUMENTATION
from IPython.utils.path import ensure_dir_exists
from IPython.utils.process import arg_split
from traitlets import (
from traitlets.config.configurable import Configurable
import __main__
class IPCompleter(Completer):
    """Extension of the completer class with IPython-specific features"""

    @observe('greedy')
    def _greedy_changed(self, change):
        """update the splitter and readline delims when greedy is changed"""
        if change['new']:
            self.evaluation = 'unsafe'
            self.auto_close_dict_keys = True
            self.splitter.delims = GREEDY_DELIMS
        else:
            self.evaluation = 'limited'
            self.auto_close_dict_keys = False
            self.splitter.delims = DELIMS
    dict_keys_only = Bool(False, help='\n        Whether to show dict key matches only.\n\n        (disables all matchers except for `IPCompleter.dict_key_matcher`).\n        ')
    suppress_competing_matchers = UnionTrait([Bool(allow_none=True), DictTrait(Bool(None, allow_none=True))], default_value=None, help="\n        Whether to suppress completions from other *Matchers*.\n\n        When set to ``None`` (default) the matchers will attempt to auto-detect\n        whether suppression of other matchers is desirable. For example, at\n        the beginning of a line followed by `%` we expect a magic completion\n        to be the only applicable option, and after ``my_dict['`` we usually\n        expect a completion with an existing dictionary key.\n\n        If you want to disable this heuristic and see completions from all matchers,\n        set ``IPCompleter.suppress_competing_matchers = False``.\n        To disable the heuristic for specific matchers provide a dictionary mapping:\n        ``IPCompleter.suppress_competing_matchers = {'IPCompleter.dict_key_matcher': False}``.\n\n        Set ``IPCompleter.suppress_competing_matchers = True`` to limit\n        completions to the set of matchers with the highest priority;\n        this is equivalent to ``IPCompleter.merge_completions`` and\n        can be beneficial for performance, but will sometimes omit relevant\n        candidates from matchers further down the priority list.\n        ").tag(config=True)
    merge_completions = Bool(True, help='Whether to merge completion results into a single list\n\n        If False, only the completion results from the first non-empty\n        completer will be returned.\n\n        As of version 8.6.0, setting the value to ``False`` is an alias for:\n        ``IPCompleter.suppress_competing_matchers = True.``.\n        ').tag(config=True)
    disable_matchers = ListTrait(Unicode(), help='List of matchers to disable.\n\n        The list should contain matcher identifiers (see :any:`completion_matcher`).\n        ').tag(config=True)
    omit__names = Enum((0, 1, 2), default_value=2, help="Instruct the completer to omit private method names\n\n        Specifically, when completing on ``object.<tab>``.\n\n        When 2 [default]: all names that start with '_' will be excluded.\n\n        When 1: all 'magic' names (``__foo__``) will be excluded.\n\n        When 0: nothing will be excluded.\n        ").tag(config=True)
    limit_to__all__ = Bool(False, help='\n        DEPRECATED as of version 5.0.\n\n        Instruct the completer to use __all__ for the completion\n\n        Specifically, when completing on ``object.<tab>``.\n\n        When True: only those names in obj.__all__ will be included.\n\n        When False [default]: the __all__ attribute is ignored\n        ').tag(config=True)
    profile_completions = Bool(default_value=False, help='If True, emit profiling data for completion subsystem using cProfile.').tag(config=True)
    profiler_output_dir = Unicode(default_value='.completion_profiles', help='Template for path at which to output profile data for completions.').tag(config=True)

    @observe('limit_to__all__')
    def _limit_to_all_changed(self, change):
        warnings.warn('`IPython.core.IPCompleter.limit_to__all__` configuration value has been deprecated since IPython 5.0, will be made to have no effects and then removed in future version of IPython.', UserWarning)

    def __init__(self, shell=None, namespace=None, global_namespace=None, config=None, **kwargs):
        """IPCompleter() -> completer

        Return a completer object.

        Parameters
        ----------
        shell
            a pointer to the ipython shell itself.  This is needed
            because this completer knows about magic functions, and those can
            only be accessed via the ipython instance.
        namespace : dict, optional
            an optional dict where completions are performed.
        global_namespace : dict, optional
            secondary optional dict for completions, to
            handle cases (such as IPython embedded inside functions) where
            both Python scopes are visible.
        config : Config
            traitlet's config object
        **kwargs
            passed to super class unmodified.
        """
        self.magic_escape = ESC_MAGIC
        self.splitter = CompletionSplitter()
        super().__init__(namespace=namespace, global_namespace=global_namespace, config=config, **kwargs)
        self.matches = []
        self.shell = shell
        self.space_name_re = re.compile('([^\\\\] )')
        self.glob = glob.glob
        term = os.environ.get('TERM', 'xterm')
        self.dumb_terminal = term in ['dumb', 'emacs']
        if sys.platform == 'win32':
            self.clean_glob = self._clean_glob_win32
        else:
            self.clean_glob = self._clean_glob
        self.docstring_sig_re = re.compile('^[\\w|\\s.]+\\(([^)]*)\\).*')
        self.docstring_kwd_re = re.compile('[\\s|\\[]*(\\w+)(?:\\s*=\\s*.*)')
        self.magic_arg_matchers = [self.magic_config_matcher, self.magic_color_matcher]
        self.custom_completers = None
        self._unicode_names = None
        self._backslash_combining_matchers = [self.latex_name_matcher, self.unicode_name_matcher, back_latex_name_matcher, back_unicode_name_matcher, self.fwd_unicode_matcher]
        if not self.backslash_combining_completions:
            for matcher in self._backslash_combining_matchers:
                self.disable_matchers.append(_get_matcher_id(matcher))
        if not self.merge_completions:
            self.suppress_competing_matchers = True

    @property
    def matchers(self) -> List[Matcher]:
        """All active matcher routines for completion"""
        if self.dict_keys_only:
            return [self.dict_key_matcher]
        if self.use_jedi:
            return [*self.custom_matchers, *self._backslash_combining_matchers, *self.magic_arg_matchers, self.custom_completer_matcher, self.magic_matcher, self._jedi_matcher, self.dict_key_matcher, self.file_matcher]
        else:
            return [*self.custom_matchers, *self._backslash_combining_matchers, *self.magic_arg_matchers, self.custom_completer_matcher, self.dict_key_matcher, self.magic_matcher, self.python_matches, self.file_matcher, self.python_func_kw_matcher]

    def all_completions(self, text: str) -> List[str]:
        """
        Wrapper around the completion methods for the benefit of emacs.
        """
        prefix = text.rpartition('.')[0]
        with provisionalcompleter():
            return ['.'.join([prefix, c.text]) if prefix and self.use_jedi else c.text for c in self.completions(text, len(text))]
        return self.complete(text)[1]

    def _clean_glob(self, text: str):
        return self.glob('%s*' % text)

    def _clean_glob_win32(self, text: str):
        return [f.replace('\\', '/') for f in self.glob('%s*' % text)]

    @context_matcher()
    def file_matcher(self, context: CompletionContext) -> SimpleMatcherResult:
        """Same as :any:`file_matches`, but adopted to new Matcher API."""
        matches = self.file_matches(context.token)
        return _convert_matcher_v1_result_to_v2(matches, type='path')

    def file_matches(self, text: str) -> List[str]:
        """Match filenames, expanding ~USER type strings.

        Most of the seemingly convoluted logic in this completer is an
        attempt to handle filenames with spaces in them.  And yet it's not
        quite perfect, because Python's readline doesn't expose all of the
        GNU readline details needed for this to be done correctly.

        For a filename with a space in it, the printed completions will be
        only the parts after what's already been typed (instead of the
        full completions, as is normally done).  I don't think with the
        current (as of Python 2.3) Python readline it's possible to do
        better.

        .. deprecated:: 8.6
            You can use :meth:`file_matcher` instead.
        """
        if text.startswith('!'):
            text = text[1:]
            text_prefix = u'!'
        else:
            text_prefix = u''
        text_until_cursor = self.text_until_cursor
        open_quotes = has_open_quotes(text_until_cursor)
        if '(' in text_until_cursor or '[' in text_until_cursor:
            lsplit = text
        else:
            try:
                lsplit = arg_split(text_until_cursor)[-1]
            except ValueError:
                if open_quotes:
                    lsplit = text_until_cursor.split(open_quotes)[-1]
                else:
                    return []
            except IndexError:
                lsplit = ''
        if not open_quotes and lsplit != protect_filename(lsplit):
            has_protectables = True
            text0, text = (text, lsplit)
        else:
            has_protectables = False
            text = os.path.expanduser(text)
        if text == '':
            return [text_prefix + protect_filename(f) for f in self.glob('*')]
        if sys.platform == 'win32':
            m0 = self.clean_glob(text)
        else:
            m0 = self.clean_glob(text.replace('\\', ''))
        if has_protectables:
            len_lsplit = len(lsplit)
            matches = [text_prefix + text0 + protect_filename(f[len_lsplit:]) for f in m0]
        elif open_quotes:
            matches = m0 if sys.platform == 'win32' else [protect_filename(f, open_quotes) for f in m0]
        else:
            matches = [text_prefix + protect_filename(f) for f in m0]
        return [x + '/' if os.path.isdir(x) else x for x in matches]

    @context_matcher()
    def magic_matcher(self, context: CompletionContext) -> SimpleMatcherResult:
        """Match magics."""
        text = context.token
        matches = self.magic_matches(text)
        result = _convert_matcher_v1_result_to_v2(matches, type='magic')
        is_magic_prefix = len(text) > 0 and text[0] == '%'
        result['suppress'] = is_magic_prefix and bool(result['completions'])
        return result

    def magic_matches(self, text: str):
        """Match magics.

        .. deprecated:: 8.6
            You can use :meth:`magic_matcher` instead.
        """
        lsm = self.shell.magics_manager.lsmagic()
        line_magics = lsm['line']
        cell_magics = lsm['cell']
        pre = self.magic_escape
        pre2 = pre + pre
        explicit_magic = text.startswith(pre)
        bare_text = text.lstrip(pre)
        global_matches = self.global_matches(bare_text)
        if not explicit_magic:

            def matches(magic):
                """
                Filter magics, in particular remove magics that match
                a name present in global namespace.
                """
                return magic.startswith(bare_text) and magic not in global_matches
        else:

            def matches(magic):
                return magic.startswith(bare_text)
        comp = [pre2 + m for m in cell_magics if matches(m)]
        if not text.startswith(pre2):
            comp += [pre + m for m in line_magics if matches(m)]
        return comp

    @context_matcher()
    def magic_config_matcher(self, context: CompletionContext) -> SimpleMatcherResult:
        """Match class names and attributes for %config magic."""
        matches = self.magic_config_matches(context.line_with_cursor)
        return _convert_matcher_v1_result_to_v2(matches, type='param')

    def magic_config_matches(self, text: str) -> List[str]:
        """Match class names and attributes for %config magic.

        .. deprecated:: 8.6
            You can use :meth:`magic_config_matcher` instead.
        """
        texts = text.strip().split()
        if len(texts) > 0 and (texts[0] == 'config' or texts[0] == '%config'):
            classes = sorted(set([c for c in self.shell.configurables if c.__class__.class_traits(config=True)]), key=lambda x: x.__class__.__name__)
            classnames = [c.__class__.__name__ for c in classes]
            if len(texts) == 1:
                return classnames
            classname_texts = texts[1].split('.')
            classname = classname_texts[0]
            classname_matches = [c for c in classnames if c.startswith(classname)]
            if texts[1].find('.') < 0:
                return classname_matches
            elif len(classname_matches) == 1 and classname_matches[0] == classname:
                cls = classes[classnames.index(classname)].__class__
                help = cls.class_get_help()
                help = re.sub(re.compile('^--', re.MULTILINE), '', help)
                return [attr.split('=')[0] for attr in help.strip().splitlines() if attr.startswith(texts[1])]
        return []

    @context_matcher()
    def magic_color_matcher(self, context: CompletionContext) -> SimpleMatcherResult:
        """Match color schemes for %colors magic."""
        matches = self.magic_color_matches(context.line_with_cursor)
        return _convert_matcher_v1_result_to_v2(matches, type='param')

    def magic_color_matches(self, text: str) -> List[str]:
        """Match color schemes for %colors magic.

        .. deprecated:: 8.6
            You can use :meth:`magic_color_matcher` instead.
        """
        texts = text.split()
        if text.endswith(' '):
            texts.append('')
        if len(texts) == 2 and (texts[0] == 'colors' or texts[0] == '%colors'):
            prefix = texts[1]
            return [color for color in InspectColors.keys() if color.startswith(prefix)]
        return []

    @context_matcher(identifier='IPCompleter.jedi_matcher')
    def _jedi_matcher(self, context: CompletionContext) -> _JediMatcherResult:
        matches = self._jedi_matches(cursor_column=context.cursor_position, cursor_line=context.cursor_line, text=context.full_text)
        return {'completions': matches, 'suppress': False}

    def _jedi_matches(self, cursor_column: int, cursor_line: int, text: str) -> Iterator[_JediCompletionLike]:
        """
        Return a list of :any:`jedi.api.Completion`\\s object from a ``text`` and
        cursor position.

        Parameters
        ----------
        cursor_column : int
            column position of the cursor in ``text``, 0-indexed.
        cursor_line : int
            line position of the cursor in ``text``, 0-indexed
        text : str
            text to complete

        Notes
        -----
        If ``IPCompleter.debug`` is ``True`` may return a :any:`_FakeJediCompletion`
        object containing a string with the Jedi debug information attached.

        .. deprecated:: 8.6
            You can use :meth:`_jedi_matcher` instead.
        """
        namespaces = [self.namespace]
        if self.global_namespace is not None:
            namespaces.append(self.global_namespace)
        completion_filter = lambda x: x
        offset = cursor_to_position(text, cursor_line, cursor_column)
        if offset:
            pre = text[offset - 1]
            if pre == '.':
                if self.omit__names == 2:
                    completion_filter = lambda c: not c.name.startswith('_')
                elif self.omit__names == 1:
                    completion_filter = lambda c: not (c.name.startswith('__') and c.name.endswith('__'))
                elif self.omit__names == 0:
                    completion_filter = lambda x: x
                else:
                    raise ValueError("Don't understand self.omit__names == {}".format(self.omit__names))
        interpreter = jedi.Interpreter(text[:offset], namespaces)
        try_jedi = True
        try:
            completing_string = False
            try:
                first_child = next((c for c in interpreter._get_module().tree_node.children if hasattr(c, 'value')))
            except StopIteration:
                pass
            else:
                completing_string = len(first_child.value) > 0 and first_child.value[0] in {"'", '"'}
            try_jedi = not completing_string
        except Exception as e:
            if self.debug:
                print('Error detecting if completing a non-finished string :', e, '|')
        if not try_jedi:
            return iter([])
        try:
            return filter(completion_filter, interpreter.complete(column=cursor_column, line=cursor_line + 1))
        except Exception as e:
            if self.debug:
                return iter([_FakeJediCompletion('Oops Jedi has crashed, please report a bug with the following:\n"""\n%s\ns"""' % e)])
            else:
                return iter([])

    @completion_matcher(api_version=1)
    def python_matches(self, text: str) -> Iterable[str]:
        """Match attributes or global python names"""
        if '.' in text:
            try:
                matches = self.attr_matches(text)
                if text.endswith('.') and self.omit__names:
                    if self.omit__names == 1:
                        no__name = lambda txt: re.match('.*\\.__.*?__', txt) is None
                    else:
                        no__name = lambda txt: re.match('\\._.*?', txt[txt.rindex('.'):]) is None
                    matches = filter(no__name, matches)
            except NameError:
                matches = []
        else:
            matches = self.global_matches(text)
        return matches

    def _default_arguments_from_docstring(self, doc):
        """Parse the first line of docstring for call signature.

        Docstring should be of the form 'min(iterable[, key=func])
'.
        It can also parse cython docstring of the form
        'Minuit.migrad(self, int ncall=10000, resume=True, int nsplit=1)'.
        """
        if doc is None:
            return []
        line = doc.lstrip().splitlines()[0]
        sig = self.docstring_sig_re.search(line)
        if sig is None:
            return []
        sig = sig.groups()[0].split(',')
        ret = []
        for s in sig:
            ret += self.docstring_kwd_re.findall(s)
        return ret

    def _default_arguments(self, obj):
        """Return the list of default arguments of obj if it is callable,
        or empty list otherwise."""
        call_obj = obj
        ret = []
        if inspect.isbuiltin(obj):
            pass
        elif not (inspect.isfunction(obj) or inspect.ismethod(obj)):
            if inspect.isclass(obj):
                ret += self._default_arguments_from_docstring(getattr(obj, '__doc__', ''))
                call_obj = getattr(obj, '__init__', None) or getattr(obj, '__new__', None)
            elif hasattr(obj, '__call__'):
                call_obj = obj.__call__
        ret += self._default_arguments_from_docstring(getattr(call_obj, '__doc__', ''))
        _keeps = (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        try:
            sig = inspect.signature(obj)
            ret.extend((k for k, v in sig.parameters.items() if v.kind in _keeps))
        except ValueError:
            pass
        return list(set(ret))

    @context_matcher()
    def python_func_kw_matcher(self, context: CompletionContext) -> SimpleMatcherResult:
        """Match named parameters (kwargs) of the last open function."""
        matches = self.python_func_kw_matches(context.token)
        return _convert_matcher_v1_result_to_v2(matches, type='param')

    def python_func_kw_matches(self, text):
        """Match named parameters (kwargs) of the last open function.

        .. deprecated:: 8.6
            You can use :meth:`python_func_kw_matcher` instead.
        """
        if '.' in text:
            return []
        try:
            regexp = self.__funcParamsRegex
        except AttributeError:
            regexp = self.__funcParamsRegex = re.compile('\n                \'.*?(?<!\\\\)\' |    # single quoted strings or\n                ".*?(?<!\\\\)" |    # double quoted strings or\n                \\w+          |    # identifier\n                \\S                # other characters\n                ', re.VERBOSE | re.DOTALL)
        tokens = regexp.findall(self.text_until_cursor)
        iterTokens = reversed(tokens)
        openPar = 0
        for token in iterTokens:
            if token == ')':
                openPar -= 1
            elif token == '(':
                openPar += 1
                if openPar > 0:
                    break
        else:
            return []
        ids = []
        isId = re.compile('\\w+$').match
        while True:
            try:
                ids.append(next(iterTokens))
                if not isId(ids[-1]):
                    ids.pop()
                    break
                if not next(iterTokens) == '.':
                    break
            except StopIteration:
                break
        usedNamedArgs = set()
        par_level = -1
        for token, next_token in zip(tokens, tokens[1:]):
            if token == '(':
                par_level += 1
            elif token == ')':
                par_level -= 1
            if par_level != 0:
                continue
            if next_token != '=':
                continue
            usedNamedArgs.add(token)
        argMatches = []
        try:
            callableObj = '.'.join(ids[::-1])
            namedArgs = self._default_arguments(eval(callableObj, self.namespace))
            for namedArg in set(namedArgs) - usedNamedArgs:
                if namedArg.startswith(text):
                    argMatches.append('%s=' % namedArg)
        except:
            pass
        return argMatches

    @staticmethod
    def _get_keys(obj: Any) -> List[Any]:
        method = get_real_method(obj, '_ipython_key_completions_')
        if method is not None:
            return method()
        if isinstance(obj, dict) or _safe_isinstance(obj, 'pandas', 'DataFrame'):
            try:
                return list(obj.keys())
            except Exception:
                return []
        elif _safe_isinstance(obj, 'pandas', 'core', 'indexing', '_LocIndexer'):
            try:
                return list(obj.obj.keys())
            except Exception:
                return []
        elif _safe_isinstance(obj, 'numpy', 'ndarray') or _safe_isinstance(obj, 'numpy', 'void'):
            return obj.dtype.names or []
        return []

    @context_matcher()
    def dict_key_matcher(self, context: CompletionContext) -> SimpleMatcherResult:
        """Match string keys in a dictionary, after e.g. ``foo[``."""
        matches = self.dict_key_matches(context.token)
        return _convert_matcher_v1_result_to_v2(matches, type='dict key', suppress_if_matches=True)

    def dict_key_matches(self, text: str) -> List[str]:
        """Match string keys in a dictionary, after e.g. ``foo[``.

        .. deprecated:: 8.6
            You can use :meth:`dict_key_matcher` instead.
        """
        if self.text_until_cursor.strip().endswith(']'):
            return []
        match = DICT_MATCHER_REGEX.search(self.text_until_cursor)
        if match is None:
            return []
        expr, prior_tuple_keys, key_prefix = match.groups()
        obj = self._evaluate_expr(expr)
        if obj is not_found:
            return []
        keys = self._get_keys(obj)
        if not keys:
            return keys
        tuple_prefix = guarded_eval(prior_tuple_keys, EvaluationContext(globals=self.global_namespace, locals=self.namespace, evaluation=self.evaluation, in_subscript=True))
        closing_quote, token_offset, matches = match_dict_keys(keys, key_prefix, self.splitter.delims, extra_prefix=tuple_prefix)
        if not matches:
            return []
        text_start = len(self.text_until_cursor) - len(text)
        if key_prefix:
            key_start = match.start(3)
            completion_start = key_start + token_offset
        else:
            key_start = completion_start = match.end()
        if text_start > key_start:
            leading = ''
        else:
            leading = text[text_start:completion_start]
        can_close_quote = False
        can_close_bracket = False
        continuation = self.line_buffer[len(self.text_until_cursor):].strip()
        if continuation.startswith(closing_quote):
            continuation = continuation[len(closing_quote):]
        else:
            can_close_quote = True
        continuation = continuation.strip()
        has_known_tuple_handling = isinstance(obj, dict)
        can_close_bracket = not continuation.startswith(']') and self.auto_close_dict_keys
        can_close_tuple_item = not continuation.startswith(',') and has_known_tuple_handling and self.auto_close_dict_keys
        can_close_quote = can_close_quote and self.auto_close_dict_keys
        if not can_close_quote and (not can_close_bracket) and closing_quote:
            return [leading + k for k in matches]
        results = []
        end_of_tuple_or_item = _DictKeyState.END_OF_TUPLE | _DictKeyState.END_OF_ITEM
        for k, state_flag in matches.items():
            result = leading + k
            if can_close_quote and closing_quote:
                result += closing_quote
            if state_flag == end_of_tuple_or_item:
                pass
            if state_flag in end_of_tuple_or_item and can_close_bracket:
                result += ']'
            if state_flag == _DictKeyState.IN_TUPLE and can_close_tuple_item:
                result += ', '
            results.append(result)
        return results

    @context_matcher()
    def unicode_name_matcher(self, context: CompletionContext):
        """Same as :any:`unicode_name_matches`, but adopted to new Matcher API."""
        fragment, matches = self.unicode_name_matches(context.text_until_cursor)
        return _convert_matcher_v1_result_to_v2(matches, type='unicode', fragment=fragment, suppress_if_matches=True)

    @staticmethod
    def unicode_name_matches(text: str) -> Tuple[str, List[str]]:
        """Match Latex-like syntax for unicode characters base
        on the name of the character.

        This does  ``\\GREEK SMALL LETTER ETA`` -> ``η``

        Works only on valid python 3 identifier, or on combining characters that
        will combine to form a valid identifier.
        """
        slashpos = text.rfind('\\')
        if slashpos > -1:
            s = text[slashpos + 1:]
            try:
                unic = unicodedata.lookup(s)
                if ('a' + unic).isidentifier():
                    return ('\\' + s, [unic])
            except KeyError:
                pass
        return ('', [])

    @context_matcher()
    def latex_name_matcher(self, context: CompletionContext):
        """Match Latex syntax for unicode characters.

        This does both ``\\alp`` -> ``\\alpha`` and ``\\alpha`` -> ``α``
        """
        fragment, matches = self.latex_matches(context.text_until_cursor)
        return _convert_matcher_v1_result_to_v2(matches, type='latex', fragment=fragment, suppress_if_matches=True)

    def latex_matches(self, text: str) -> Tuple[str, Sequence[str]]:
        """Match Latex syntax for unicode characters.

        This does both ``\\alp`` -> ``\\alpha`` and ``\\alpha`` -> ``α``

        .. deprecated:: 8.6
            You can use :meth:`latex_name_matcher` instead.
        """
        slashpos = text.rfind('\\')
        if slashpos > -1:
            s = text[slashpos:]
            if s in latex_symbols:
                return (s, [latex_symbols[s]])
            else:
                matches = [k for k in latex_symbols if k.startswith(s)]
                if matches:
                    return (s, matches)
        return ('', ())

    @context_matcher()
    def custom_completer_matcher(self, context):
        """Dispatch custom completer.

        If a match is found, suppresses all other matchers except for Jedi.
        """
        matches = self.dispatch_custom_completer(context.token) or []
        result = _convert_matcher_v1_result_to_v2(matches, type=_UNKNOWN_TYPE, suppress_if_matches=True)
        result['ordered'] = True
        result['do_not_suppress'] = {_get_matcher_id(self._jedi_matcher)}
        return result

    def dispatch_custom_completer(self, text):
        """
        .. deprecated:: 8.6
            You can use :meth:`custom_completer_matcher` instead.
        """
        if not self.custom_completers:
            return
        line = self.line_buffer
        if not line.strip():
            return None
        event = SimpleNamespace()
        event.line = line
        event.symbol = text
        cmd = line.split(None, 1)[0]
        event.command = cmd
        event.text_until_cursor = self.text_until_cursor
        if not cmd.startswith(self.magic_escape):
            try_magic = self.custom_completers.s_matches(self.magic_escape + cmd)
        else:
            try_magic = []
        for c in itertools.chain(self.custom_completers.s_matches(cmd), try_magic, self.custom_completers.flat_matches(self.text_until_cursor)):
            try:
                res = c(event)
                if res:
                    withcase = [r for r in res if r.startswith(text)]
                    if withcase:
                        return withcase
                    text_low = text.lower()
                    return [r for r in res if r.lower().startswith(text_low)]
            except TryNext:
                pass
            except KeyboardInterrupt:
                '\n                If custom completer take too long,\n                let keyboard interrupt abort and return nothing.\n                '
                break
        return None

    def completions(self, text: str, offset: int) -> Iterator[Completion]:
        """
        Returns an iterator over the possible completions

        .. warning::

            Unstable

            This function is unstable, API may change without warning.
            It will also raise unless use in proper context manager.

        Parameters
        ----------
        text : str
            Full text of the current input, multi line string.
        offset : int
            Integer representing the position of the cursor in ``text``. Offset
            is 0-based indexed.

        Yields
        ------
        Completion

        Notes
        -----
        The cursor on a text can either be seen as being "in between"
        characters or "On" a character depending on the interface visible to
        the user. For consistency the cursor being on "in between" characters X
        and Y is equivalent to the cursor being "on" character Y, that is to say
        the character the cursor is on is considered as being after the cursor.

        Combining characters may span more that one position in the
        text.

        .. note::

            If ``IPCompleter.debug`` is :any:`True` will yield a ``--jedi/ipython--``
            fake Completion token to distinguish completion returned by Jedi
            and usual IPython completion.

        .. note::

            Completions are not completely deduplicated yet. If identical
            completions are coming from different sources this function does not
            ensure that each completion object will only be present once.
        """
        warnings.warn('_complete is a provisional API (as of IPython 6.0). It may change without warnings. Use in corresponding context manager.', category=ProvisionalCompleterWarning, stacklevel=2)
        seen = set()
        profiler: Optional[cProfile.Profile]
        try:
            if self.profile_completions:
                import cProfile
                profiler = cProfile.Profile()
                profiler.enable()
            else:
                profiler = None
            for c in self._completions(text, offset, _timeout=self.jedi_compute_type_timeout / 1000):
                if c and c in seen:
                    continue
                yield c
                seen.add(c)
        except KeyboardInterrupt:
            'if completions take too long and users send keyboard interrupt,\n            do not crash and return ASAP. '
            pass
        finally:
            if profiler is not None:
                profiler.disable()
                ensure_dir_exists(self.profiler_output_dir)
                output_path = os.path.join(self.profiler_output_dir, str(uuid.uuid4()))
                print('Writing profiler output to', output_path)
                profiler.dump_stats(output_path)

    def _completions(self, full_text: str, offset: int, *, _timeout) -> Iterator[Completion]:
        """
        Core completion module.Same signature as :any:`completions`, with the
        extra `timeout` parameter (in seconds).

        Computing jedi's completion ``.type`` can be quite expensive (it is a
        lazy property) and can require some warm-up, more warm up than just
        computing the ``name`` of a completion. The warm-up can be :

            - Long warm-up the first time a module is encountered after
            install/update: actually build parse/inference tree.

            - first time the module is encountered in a session: load tree from
            disk.

        We don't want to block completions for tens of seconds so we give the
        completer a "budget" of ``_timeout`` seconds per invocation to compute
        completions types, the completions that have not yet been computed will
        be marked as "unknown" an will have a chance to be computed next round
        are things get cached.

        Keep in mind that Jedi is not the only thing treating the completion so
        keep the timeout short-ish as if we take more than 0.3 second we still
        have lots of processing to do.

        """
        deadline = time.monotonic() + _timeout
        before = full_text[:offset]
        cursor_line, cursor_column = position_to_cursor(full_text, offset)
        jedi_matcher_id = _get_matcher_id(self._jedi_matcher)

        def is_non_jedi_result(result: MatcherResult, identifier: str) -> TypeGuard[SimpleMatcherResult]:
            return identifier != jedi_matcher_id
        results = self._complete(full_text=full_text, cursor_line=cursor_line, cursor_pos=cursor_column)
        non_jedi_results: Dict[str, SimpleMatcherResult] = {identifier: result for identifier, result in results.items() if is_non_jedi_result(result, identifier)}
        jedi_matches = cast(_JediMatcherResult, results[jedi_matcher_id])['completions'] if jedi_matcher_id in results else ()
        iter_jm = iter(jedi_matches)
        if _timeout:
            for jm in iter_jm:
                try:
                    type_ = jm.type
                except Exception:
                    if self.debug:
                        print('Error in Jedi getting type of ', jm)
                    type_ = None
                delta = len(jm.name_with_symbols) - len(jm.complete)
                if type_ == 'function':
                    signature = _make_signature(jm)
                else:
                    signature = ''
                yield Completion(start=offset - delta, end=offset, text=jm.name_with_symbols, type=type_, signature=signature, _origin='jedi')
                if time.monotonic() > deadline:
                    break
        for jm in iter_jm:
            delta = len(jm.name_with_symbols) - len(jm.complete)
            yield Completion(start=offset - delta, end=offset, text=jm.name_with_symbols, type=_UNKNOWN_TYPE, _origin='jedi', signature='')
        if jedi_matches and non_jedi_results and self.debug:
            some_start_offset = before.rfind(next(iter(non_jedi_results.values()))['matched_fragment'])
            yield Completion(start=some_start_offset, end=offset, text='--jedi/ipython--', _origin='debug', type='none', signature='')
        ordered: List[Completion] = []
        sortable: List[Completion] = []
        for origin, result in non_jedi_results.items():
            matched_text = result['matched_fragment']
            start_offset = before.rfind(matched_text)
            is_ordered = result.get('ordered', False)
            container = ordered if is_ordered else sortable
            assert before.endswith(matched_text)
            for simple_completion in result['completions']:
                completion = Completion(start=start_offset, end=offset, text=simple_completion.text, _origin=origin, signature='', type=simple_completion.type or _UNKNOWN_TYPE)
                container.append(completion)
        yield from list(self._deduplicate(ordered + self._sort(sortable)))[:MATCHES_LIMIT]

    def complete(self, text=None, line_buffer=None, cursor_pos=None) -> Tuple[str, Sequence[str]]:
        """Find completions for the given text and line context.

        Note that both the text and the line_buffer are optional, but at least
        one of them must be given.

        Parameters
        ----------
        text : string, optional
            Text to perform the completion on.  If not given, the line buffer
            is split using the instance's CompletionSplitter object.
        line_buffer : string, optional
            If not given, the completer attempts to obtain the current line
            buffer via readline.  This keyword allows clients which are
            requesting for text completions in non-readline contexts to inform
            the completer of the entire text.
        cursor_pos : int, optional
            Index of the cursor in the full line buffer.  Should be provided by
            remote frontends where kernel has no access to frontend state.

        Returns
        -------
        Tuple of two items:
        text : str
            Text that was actually used in the completion.
        matches : list
            A list of completion matches.

        Notes
        -----
            This API is likely to be deprecated and replaced by
            :any:`IPCompleter.completions` in the future.

        """
        warnings.warn('`Completer.complete` is pending deprecation since IPython 6.0 and will be replaced by `Completer.completions`.', PendingDeprecationWarning)
        results = self._complete(line_buffer=line_buffer, cursor_pos=cursor_pos, text=text, cursor_line=0)
        jedi_matcher_id = _get_matcher_id(self._jedi_matcher)
        return self._arrange_and_extract(results, skip_matchers={jedi_matcher_id}, abort_if_offset_changes=True)

    def _arrange_and_extract(self, results: Dict[str, MatcherResult], skip_matchers: Set[str], abort_if_offset_changes: bool):
        sortable: List[AnyMatcherCompletion] = []
        ordered: List[AnyMatcherCompletion] = []
        most_recent_fragment = None
        for identifier, result in results.items():
            if identifier in skip_matchers:
                continue
            if not result['completions']:
                continue
            if not most_recent_fragment:
                most_recent_fragment = result['matched_fragment']
            if abort_if_offset_changes and result['matched_fragment'] != most_recent_fragment:
                break
            if result.get('ordered', False):
                ordered.extend(result['completions'])
            else:
                sortable.extend(result['completions'])
        if not most_recent_fragment:
            most_recent_fragment = ''
        return (most_recent_fragment, [m.text for m in self._deduplicate(ordered + self._sort(sortable))])

    def _complete(self, *, cursor_line, cursor_pos, line_buffer=None, text=None, full_text=None) -> _CompleteResult:
        """
        Like complete but can also returns raw jedi completions as well as the
        origin of the completion text. This could (and should) be made much
        cleaner but that will be simpler once we drop the old (and stateful)
        :any:`complete` API.

        With current provisional API, cursor_pos act both (depending on the
        caller) as the offset in the ``text`` or ``line_buffer``, or as the
        ``column`` when passing multiline strings this could/should be renamed
        but would add extra noise.

        Parameters
        ----------
        cursor_line
            Index of the line the cursor is on. 0 indexed.
        cursor_pos
            Position of the cursor in the current line/line_buffer/text. 0
            indexed.
        line_buffer : optional, str
            The current line the cursor is in, this is mostly due to legacy
            reason that readline could only give a us the single current line.
            Prefer `full_text`.
        text : str
            The current "token" the cursor is in, mostly also for historical
            reasons. as the completer would trigger only after the current line
            was parsed.
        full_text : str
            Full text of the current cell.

        Returns
        -------
        An ordered dictionary where keys are identifiers of completion
        matchers and values are ``MatcherResult``s.
        """
        if cursor_pos is None:
            cursor_pos = len(line_buffer) if text is None else len(text)
        if self.use_main_ns:
            self.namespace = __main__.__dict__
        if not line_buffer and full_text:
            line_buffer = full_text.split('\n')[cursor_line]
        if not text:
            text = self.splitter.split_line(line_buffer, cursor_pos) if line_buffer else ''
        if line_buffer is None:
            line_buffer = text
        self.line_buffer = line_buffer
        self.text_until_cursor = self.line_buffer[:cursor_pos]
        if not full_text:
            full_text = line_buffer
        context = CompletionContext(full_text=full_text, cursor_position=cursor_pos, cursor_line=cursor_line, token=text, limit=MATCHES_LIMIT)
        results: Dict[str, MatcherResult] = {}
        jedi_matcher_id = _get_matcher_id(self._jedi_matcher)
        suppressed_matchers: Set[str] = set()
        matchers = {_get_matcher_id(matcher): matcher for matcher in sorted(self.matchers, key=_get_matcher_priority, reverse=True)}
        for matcher_id, matcher in matchers.items():
            matcher_id = _get_matcher_id(matcher)
            if matcher_id in self.disable_matchers:
                continue
            if matcher_id in results:
                warnings.warn(f'Duplicate matcher ID: {matcher_id}.')
            if matcher_id in suppressed_matchers:
                continue
            result: MatcherResult
            try:
                if _is_matcher_v1(matcher):
                    result = _convert_matcher_v1_result_to_v2(matcher(text), type=_UNKNOWN_TYPE)
                elif _is_matcher_v2(matcher):
                    result = matcher(context)
                else:
                    api_version = _get_matcher_api_version(matcher)
                    raise ValueError(f'Unsupported API version {api_version}')
            except:
                sys.excepthook(*sys.exc_info())
                continue
            result['matched_fragment'] = result.get('matched_fragment', context.token)
            if not suppressed_matchers:
                suppression_recommended: Union[bool, Set[str]] = result.get('suppress', False)
                suppression_config = self.suppress_competing_matchers.get(matcher_id, None) if isinstance(self.suppress_competing_matchers, dict) else self.suppress_competing_matchers
                should_suppress = (suppression_config is True or (suppression_recommended and suppression_config is not False)) and has_any_completions(result)
                if should_suppress:
                    suppression_exceptions: Set[str] = result.get('do_not_suppress', set())
                    if isinstance(suppression_recommended, Iterable):
                        to_suppress = set(suppression_recommended)
                    else:
                        to_suppress = set(matchers)
                    suppressed_matchers = to_suppress - suppression_exceptions
                    new_results = {}
                    for previous_matcher_id, previous_result in results.items():
                        if previous_matcher_id not in suppressed_matchers:
                            new_results[previous_matcher_id] = previous_result
                    results = new_results
            results[matcher_id] = result
        _, matches = self._arrange_and_extract(results, skip_matchers={jedi_matcher_id}, abort_if_offset_changes=False)
        self.matches = matches
        return results

    @staticmethod
    def _deduplicate(matches: Sequence[AnyCompletion]) -> Iterable[AnyCompletion]:
        filtered_matches: Dict[str, AnyCompletion] = {}
        for match in matches:
            text = match.text
            if text not in filtered_matches or filtered_matches[text].type == _UNKNOWN_TYPE:
                filtered_matches[text] = match
        return filtered_matches.values()

    @staticmethod
    def _sort(matches: Sequence[AnyCompletion]):
        return sorted(matches, key=lambda x: completions_sorting_key(x.text))

    @context_matcher()
    def fwd_unicode_matcher(self, context: CompletionContext):
        """Same as :any:`fwd_unicode_match`, but adopted to new Matcher API."""
        fragment, matches = self.fwd_unicode_match(context.text_until_cursor)
        return _convert_matcher_v1_result_to_v2(matches, type='unicode', fragment=fragment, suppress_if_matches=True)

    def fwd_unicode_match(self, text: str) -> Tuple[str, Sequence[str]]:
        """
        Forward match a string starting with a backslash with a list of
        potential Unicode completions.

        Will compute list of Unicode character names on first call and cache it.

        .. deprecated:: 8.6
            You can use :meth:`fwd_unicode_matcher` instead.

        Returns
        -------
        At tuple with:
            - matched text (empty if no matches)
            - list of potential completions, empty tuple  otherwise)
        """
        slashpos = text.rfind('\\')
        if slashpos > -1:
            s = text[slashpos + 1:]
            sup = s.upper()
            candidates = [x for x in self.unicode_names if x.startswith(sup)]
            if candidates:
                return (s, candidates)
            candidates = [x for x in self.unicode_names if sup in x]
            if candidates:
                return (s, candidates)
            splitsup = sup.split(' ')
            candidates = [x for x in self.unicode_names if all((u in x for u in splitsup))]
            if candidates:
                return (s, candidates)
            return ('', ())
        else:
            return ('', ())

    @property
    def unicode_names(self) -> List[str]:
        """List of names of unicode code points that can be completed.

        The list is lazily initialized on first access.
        """
        if self._unicode_names is None:
            names = []
            for c in range(0, 1114111 + 1):
                try:
                    names.append(unicodedata.name(chr(c)))
                except ValueError:
                    pass
            self._unicode_names = _unicode_name_compute(_UNICODE_RANGES)
        return self._unicode_names