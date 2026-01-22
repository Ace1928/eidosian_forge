import functools
import re
import nltk.tree
def _tgrep_conjunction_action(_s, _l, tokens, join_char='&'):
    """
    Builds a lambda function representing a predicate on a tree node
    from the conjunction of several other such lambda functions.

    This is prototypically called for expressions like
    (`tgrep_rel_conjunction`)::

        < NP & < AP < VP

    where tokens is a list of predicates representing the relations
    (`< NP`, `< AP`, and `< VP`), possibly with the character `&`
    included (as in the example here).

    This is also called for expressions like (`tgrep_node_expr2`)::

        NP < NN
        S=s < /NP/=n : s < /VP/=v : n .. v

    tokens[0] is a tgrep_expr predicate; tokens[1:] are an (optional)
    list of segmented patterns (`tgrep_expr_labeled`, processed by
    `_tgrep_segmented_pattern_action`).
    """
    tokens = [x for x in tokens if x != join_char]
    if len(tokens) == 1:
        return tokens[0]
    else:
        return (lambda ts: lambda n, m=None, l=None: all((predicate(n, m, l) for predicate in ts)))(tokens)