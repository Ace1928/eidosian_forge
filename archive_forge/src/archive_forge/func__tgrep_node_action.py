import functools
import re
import nltk.tree
def _tgrep_node_action(_s, _l, tokens):
    """
    Builds a lambda function representing a predicate on a tree node
    depending on the name of its node.
    """
    if tokens[0] == "'":
        tokens = tokens[1:]
    if len(tokens) > 1:
        assert list(set(tokens[1::2])) == ['|']
        tokens = [_tgrep_node_action(None, None, [node]) for node in tokens[::2]]
        return (lambda t: lambda n, m=None, l=None: any((f(n, m, l) for f in t)))(tokens)
    elif hasattr(tokens[0], '__call__'):
        return tokens[0]
    elif tokens[0] == '*' or tokens[0] == '__':
        return lambda n, m=None, l=None: True
    elif tokens[0].startswith('"'):
        assert tokens[0].endswith('"')
        node_lit = tokens[0][1:-1].replace('\\"', '"').replace('\\\\', '\\')
        return (lambda s: lambda n, m=None, l=None: _tgrep_node_literal_value(n) == s)(node_lit)
    elif tokens[0].startswith('/'):
        assert tokens[0].endswith('/')
        node_lit = tokens[0][1:-1]
        return (lambda r: lambda n, m=None, l=None: r.search(_tgrep_node_literal_value(n)))(re.compile(node_lit))
    elif tokens[0].startswith('i@'):
        node_func = _tgrep_node_action(_s, _l, [tokens[0][2:].lower()])
        return (lambda f: lambda n, m=None, l=None: f(_tgrep_node_literal_value(n).lower()))(node_func)
    else:
        return (lambda s: lambda n, m=None, l=None: _tgrep_node_literal_value(n) == s)(tokens[0])