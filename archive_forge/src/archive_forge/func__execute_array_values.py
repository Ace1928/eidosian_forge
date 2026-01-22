import re
import warnings
from parso import parse, ParserSyntaxError
from jedi import debug
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import iterator_to_value_set, ValueSet, \
from jedi.inference.lazy_value import LazyKnownValues
def _execute_array_values(inference_state, array):
    """
    Tuples indicate that there's not just one return value, but the listed
    ones.  `(str, int)` means that it returns a tuple with both types.
    """
    from jedi.inference.value.iterable import SequenceLiteralValue, FakeTuple, FakeList
    if isinstance(array, SequenceLiteralValue) and array.array_type in ('tuple', 'list'):
        values = []
        for lazy_value in array.py__iter__():
            objects = ValueSet.from_sets((_execute_array_values(inference_state, typ) for typ in lazy_value.infer()))
            values.append(LazyKnownValues(objects))
        cls = FakeTuple if array.array_type == 'tuple' else FakeList
        return {cls(inference_state, values)}
    else:
        return array.execute_annotation()