from jedi import debug
from jedi.cache import memoize_method
from jedi.inference.utils import to_tuple
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.value.iterable import SequenceLiteralValue
from jedi.inference.helpers import is_string
class TupleGenericManager(_AbstractGenericManager):

    def __init__(self, tup):
        self._tuple = tup

    def __getitem__(self, index):
        return self._tuple[index]

    def __len__(self):
        return len(self._tuple)

    def to_tuple(self):
        return self._tuple

    def is_homogenous_tuple(self):
        return False

    def __repr__(self):
        return '<TupG>[%s]' % ', '.join((repr(x) for x in self.to_tuple()))