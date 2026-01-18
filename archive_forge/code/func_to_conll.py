import subprocess
import warnings
from collections import defaultdict
from itertools import chain
from pprint import pformat
from nltk.internals import find_binary
from nltk.tree import Tree
def to_conll(self, style):
    """
        The dependency graph in CoNLL format.

        :param style: the style to use for the format (3, 4, 10 columns)
        :type style: int
        :rtype: str
        """
    if style == 3:
        template = '{word}\t{tag}\t{head}\n'
    elif style == 4:
        template = '{word}\t{tag}\t{head}\t{rel}\n'
    elif style == 10:
        template = '{i}\t{word}\t{lemma}\t{ctag}\t{tag}\t{feats}\t{head}\t{rel}\t_\t_\n'
    else:
        raise ValueError('Number of tab-delimited fields ({}) not supported by CoNLL(10) or Malt-Tab(4) format'.format(style))
    return ''.join((template.format(i=i, **node) for i, node in sorted(self.nodes.items()) if node['tag'] != 'TOP'))