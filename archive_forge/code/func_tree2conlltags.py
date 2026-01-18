import re
from nltk.metrics import accuracy as _accuracy
from nltk.tag.mapping import map_tag
from nltk.tag.util import str2tuple
from nltk.tree import Tree
def tree2conlltags(t):
    """
    Return a list of 3-tuples containing ``(word, tag, IOB-tag)``.
    Convert a tree to the CoNLL IOB tag format.

    :param t: The tree to be converted.
    :type t: Tree
    :rtype: list(tuple)
    """
    tags = []
    for child in t:
        try:
            category = child.label()
            prefix = 'B-'
            for contents in child:
                if isinstance(contents, Tree):
                    raise ValueError('Tree is too deeply nested to be printed in CoNLL format')
                tags.append((contents[0], contents[1], prefix + category))
                prefix = 'I-'
        except AttributeError:
            tags.append((child[0], child[1], 'O'))
    return tags