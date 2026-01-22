import textwrap
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import Tree
from nltk.util import LazyConcatenation, LazyMap
def _get_srl_instances(self, grid, pos_in_tree):
    tree = self._get_parsed_sent(grid, pos_in_tree)
    spanlists = self._get_srl_spans(grid)
    if self._srl_includes_roleset:
        predicates = self._get_column(grid, self._colmap['srl'] + 1)
        rolesets = self._get_column(grid, self._colmap['srl'])
    else:
        predicates = self._get_column(grid, self._colmap['srl'])
        rolesets = [None] * len(predicates)
    instances = ConllSRLInstanceList(tree)
    for wordnum, predicate in enumerate(predicates):
        if predicate == '-':
            continue
        for spanlist in spanlists:
            for (start, end), tag in spanlist:
                if wordnum in range(start, end) and tag in ('V', 'C-V'):
                    break
            else:
                continue
            break
        else:
            raise ValueError('No srl column found for %r' % predicate)
        instances.append(ConllSRLInstance(tree, wordnum, predicate, rolesets[wordnum], spanlist))
    return instances