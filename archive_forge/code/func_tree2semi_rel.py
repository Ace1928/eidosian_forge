import html
import re
from collections import defaultdict
def tree2semi_rel(tree):
    """
    Group a chunk structure into a list of 'semi-relations' of the form (list(str), ``Tree``).

    In order to facilitate the construction of (``Tree``, string, ``Tree``) triples, this
    identifies pairs whose first member is a list (possibly empty) of terminal
    strings, and whose second member is a ``Tree`` of the form (NE_label, terminals).

    :param tree: a chunk tree
    :return: a list of pairs (list(str), ``Tree``)
    :rtype: list of tuple
    """
    from nltk.tree import Tree
    semi_rels = []
    semi_rel = [[], None]
    for dtr in tree:
        if not isinstance(dtr, Tree):
            semi_rel[0].append(dtr)
        else:
            semi_rel[1] = dtr
            semi_rels.append(semi_rel)
            semi_rel = [[], None]
    return semi_rels