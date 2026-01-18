import re
from nltk.metrics import accuracy as _accuracy
from nltk.tag.mapping import map_tag
from nltk.tag.util import str2tuple
from nltk.tree import Tree
def tree2conllstr(t):
    """
    Return a multiline string where each line contains a word, tag and IOB tag.
    Convert a tree to the CoNLL IOB string format

    :param t: The tree to be converted.
    :type t: Tree
    :rtype: str
    """
    lines = [' '.join(token) for token in tree2conlltags(t)]
    return '\n'.join(lines)