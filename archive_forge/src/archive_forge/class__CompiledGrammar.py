from __future__ import unicode_literals
import re
from six.moves import range
from .regex_parser import Any, Sequence, Regex, Variable, Repeat, Lookahead
from .regex_parser import parse_regex, tokenize_regex
class _CompiledGrammar(object):
    """
    Compiles a grammar. This will take the parse tree of a regular expression
    and compile the grammar.

    :param root_node: :class~`.regex_parser.Node` instance.
    :param escape_funcs: `dict` mapping variable names to escape callables.
    :param unescape_funcs: `dict` mapping variable names to unescape callables.
    """

    def __init__(self, root_node, escape_funcs=None, unescape_funcs=None):
        self.root_node = root_node
        self.escape_funcs = escape_funcs or {}
        self.unescape_funcs = unescape_funcs or {}
        self._group_names_to_nodes = {}
        counter = [0]

        def create_group_func(node):
            name = 'n%s' % counter[0]
            self._group_names_to_nodes[name] = node.varname
            counter[0] += 1
            return name
        self._re_pattern = '^%s$' % self._transform(root_node, create_group_func)
        self._re_prefix_patterns = list(self._transform_prefix(root_node, create_group_func))
        flags = re.DOTALL
        self._re = re.compile(self._re_pattern, flags)
        self._re_prefix = [re.compile(t, flags) for t in self._re_prefix_patterns]
        self._re_prefix_with_trailing_input = [re.compile('(?:%s)(?P<%s>.*?)$' % (t.rstrip('$'), _INVALID_TRAILING_INPUT), flags) for t in self._re_prefix_patterns]

    def escape(self, varname, value):
        """
        Escape `value` to fit in the place of this variable into the grammar.
        """
        f = self.escape_funcs.get(varname)
        return f(value) if f else value

    def unescape(self, varname, value):
        """
        Unescape `value`.
        """
        f = self.unescape_funcs.get(varname)
        return f(value) if f else value

    @classmethod
    def _transform(cls, root_node, create_group_func):
        """
        Turn a :class:`Node` object into a regular expression.

        :param root_node: The :class:`Node` instance for which we generate the grammar.
        :param create_group_func: A callable which takes a `Node` and returns the next
            free name for this node.
        """

        def transform(node):
            if isinstance(node, Any):
                return '(?:%s)' % '|'.join((transform(c) for c in node.children))
            elif isinstance(node, Sequence):
                return ''.join((transform(c) for c in node.children))
            elif isinstance(node, Regex):
                return node.regex
            elif isinstance(node, Lookahead):
                before = '(?!' if node.negative else '(='
                return before + transform(node.childnode) + ')'
            elif isinstance(node, Variable):
                return '(?P<%s>%s)' % (create_group_func(node), transform(node.childnode))
            elif isinstance(node, Repeat):
                return '(?:%s){%i,%s}%s' % (transform(node.childnode), node.min_repeat, '' if node.max_repeat is None else str(node.max_repeat), '' if node.greedy else '?')
            else:
                raise TypeError('Got %r' % (node,))
        return transform(root_node)

    @classmethod
    def _transform_prefix(cls, root_node, create_group_func):
        """
        Yield all the regular expressions matching a prefix of the grammar
        defined by the `Node` instance.

        This can yield multiple expressions, because in the case of on OR
        operation in the grammar, we can have another outcome depending on
        which clause would appear first. E.g. "(A|B)C" is not the same as
        "(B|A)C" because the regex engine is lazy and takes the first match.
        However, because we the current input is actually a prefix of the
        grammar which meight not yet contain the data for "C", we need to know
        both intermediate states, in order to call the appropriate
        autocompletion for both cases.

        :param root_node: The :class:`Node` instance for which we generate the grammar.
        :param create_group_func: A callable which takes a `Node` and returns the next
            free name for this node.
        """

        def transform(node):
            if isinstance(node, Any):
                for c in node.children:
                    for r in transform(c):
                        yield ('(?:%s)?' % r)
            elif isinstance(node, Sequence):
                for i in range(len(node.children)):
                    a = [cls._transform(c, create_group_func) for c in node.children[:i]]
                    for c in transform(node.children[i]):
                        yield ('(?:%s)' % (''.join(a) + c))
            elif isinstance(node, Regex):
                yield ('(?:%s)?' % node.regex)
            elif isinstance(node, Lookahead):
                if node.negative:
                    yield ('(?!%s)' % cls._transform(node.childnode, create_group_func))
                else:
                    raise Exception('Positive lookahead not yet supported.')
            elif isinstance(node, Variable):
                for c in transform(node.childnode):
                    yield ('(?P<%s>%s)' % (create_group_func(node), c))
            elif isinstance(node, Repeat):
                prefix = cls._transform(node.childnode, create_group_func)
                for c in transform(node.childnode):
                    if node.max_repeat:
                        repeat_sign = '{,%i}' % (node.max_repeat - 1)
                    else:
                        repeat_sign = '*'
                    yield ('(?:%s)%s%s(?:%s)?' % (prefix, repeat_sign, '' if node.greedy else '?', c))
            else:
                raise TypeError('Got %r' % node)
        for r in transform(root_node):
            yield ('^%s$' % r)

    def match(self, string):
        """
        Match the string with the grammar.
        Returns a :class:`Match` instance or `None` when the input doesn't match the grammar.

        :param string: The input string.
        """
        m = self._re.match(string)
        if m:
            return Match(string, [(self._re, m)], self._group_names_to_nodes, self.unescape_funcs)

    def match_prefix(self, string):
        """
        Do a partial match of the string with the grammar. The returned
        :class:`Match` instance can contain multiple representations of the
        match. This will never return `None`. If it doesn't match at all, the "trailing input"
        part will capture all of the input.

        :param string: The input string.
        """
        for patterns in [self._re_prefix, self._re_prefix_with_trailing_input]:
            matches = [(r, r.match(string)) for r in patterns]
            matches = [(r, m) for r, m in matches if m]
            if matches != []:
                return Match(string, matches, self._group_names_to_nodes, self.unescape_funcs)