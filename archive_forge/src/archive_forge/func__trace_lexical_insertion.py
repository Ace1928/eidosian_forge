from functools import reduce
from nltk.parse.api import ParserI
from nltk.tree import ProbabilisticTree, Tree
def _trace_lexical_insertion(self, token, index, width):
    str = '   Insert: |' + '.' * index + '=' + '.' * (width - index - 1) + '| '
    str += f'{token}'
    print(str)