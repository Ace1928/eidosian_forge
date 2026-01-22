from jedi import debug
from jedi import settings
from jedi.inference import recursion
from jedi.inference.base_value import ValueSet, NO_VALUES, HelperValueMixin, \
from jedi.inference.lazy_value import LazyKnownValues
from jedi.inference.helpers import infer_call_of_leaf
from jedi.inference.cache import inference_state_method_cache
class _Modification(ValueWrapper):

    def __init__(self, wrapped_value, assigned_values, contextualized_key):
        super().__init__(wrapped_value)
        self._assigned_values = assigned_values
        self._contextualized_key = contextualized_key

    def py__getitem__(self, *args, **kwargs):
        return self._wrapped_value.py__getitem__(*args, **kwargs) | self._assigned_values

    def py__simple_getitem__(self, index):
        actual = [v.get_safe_value(_sentinel) for v in self._contextualized_key.infer()]
        if index in actual:
            return self._assigned_values
        return self._wrapped_value.py__simple_getitem__(index)