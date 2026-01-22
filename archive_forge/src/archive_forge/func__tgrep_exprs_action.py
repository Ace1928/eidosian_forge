import functools
import re
import nltk.tree
def _tgrep_exprs_action(_s, _l, tokens):
    """
    This is the top-lebel node in a tgrep2 search string; the
    predicate function it returns binds together all the state of a
    tgrep2 search string.

    Builds a lambda function representing a predicate on a tree node
    from the disjunction of several tgrep expressions.  Also handles
    macro definitions and macro name binding, and node label
    definitions and node label binding.
    """
    if len(tokens) == 1:
        return lambda n, m=None, l=None: tokens[0](n, None, {})
    tokens = [x for x in tokens if x != ';']
    macro_dict = {}
    macro_defs = [tok for tok in tokens if isinstance(tok, dict)]
    for macro_def in macro_defs:
        macro_dict.update(macro_def)
    tgrep_exprs = [tok for tok in tokens if not isinstance(tok, dict)]

    def top_level_pred(n, m=macro_dict, l=None):
        label_dict = {}
        return any((predicate(n, m, label_dict) for predicate in tgrep_exprs))
    return top_level_pred