import functools
import re
import nltk.tree
def _tgrep_rel_disjunction_action(_s, _l, tokens):
    """
    Builds a lambda function representing a predicate on a tree node
    from the disjunction of several other such lambda functions.
    """
    tokens = [x for x in tokens if x != '|']
    if len(tokens) == 1:
        return tokens[0]
    elif len(tokens) == 2:
        return (lambda a, b: lambda n, m=None, l=None: a(n, m, l) or b(n, m, l))(tokens[0], tokens[1])