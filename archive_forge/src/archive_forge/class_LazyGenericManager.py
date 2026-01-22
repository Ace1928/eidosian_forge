from jedi import debug
from jedi.cache import memoize_method
from jedi.inference.utils import to_tuple
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.value.iterable import SequenceLiteralValue
from jedi.inference.helpers import is_string
class LazyGenericManager(_AbstractGenericManager):

    def __init__(self, context_of_index, index_value):
        self._context_of_index = context_of_index
        self._index_value = index_value

    @memoize_method
    def __getitem__(self, index):
        return self._tuple()[index]()

    def __len__(self):
        return len(self._tuple())

    @memoize_method
    @to_tuple
    def _tuple(self):

        def lambda_scoping_in_for_loop_sucks(lazy_value):
            return lambda: ValueSet(_resolve_forward_references(self._context_of_index, lazy_value.infer()))
        if isinstance(self._index_value, SequenceLiteralValue):
            for lazy_value in self._index_value.py__iter__(contextualized_node=None):
                yield lambda_scoping_in_for_loop_sucks(lazy_value)
        else:
            yield (lambda: ValueSet(_resolve_forward_references(self._context_of_index, ValueSet([self._index_value]))))

    @to_tuple
    def to_tuple(self):
        for callable_ in self._tuple():
            yield callable_()

    def is_homogenous_tuple(self):
        if isinstance(self._index_value, SequenceLiteralValue):
            entries = self._index_value.get_tree_entries()
            if len(entries) == 2 and entries[1] == '...':
                return True
        return False

    def __repr__(self):
        return '<LazyG>[%s]' % ', '.join((repr(x) for x in self.to_tuple()))