import itertools
from typing import Any, Callable, Dict, Set
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import annos
class _TypeMap(object):
    """Abstraction for the state of the CFG walk for type inference.

  This is a value type. Only implements the strictly necessary operators.

  Attributes:
    types: Dict[qual_names.QN, Set[Type]], mapping symbols to the set of
      possible types.
  """

    def __init__(self, init_from=None):
        if init_from:
            assert isinstance(init_from, _TypeMap)
            self.types = {s: set(other_types) for s, other_types in init_from.types.items()}
        else:
            self.types = {}

    def __eq__(self, other):
        if frozenset(self.types.keys()) != frozenset(other.types.keys()):
            return False
        ret = all((self.types[s] == other.types[s] for s in self.types))
        return ret

    def __ne__(self, other):
        return not self.__eq__(other)

    def __or__(self, other):
        assert isinstance(other, _TypeMap)
        result = _TypeMap(self)
        for s, other_types in other.types.items():
            if s not in result.types:
                self_types = set()
                result.types[s] = self_types
            else:
                self_types = result.types[s]
            self_types.update(other_types)
        return result

    def __repr__(self):
        return 'SymbolTable {}'.format(self.types)