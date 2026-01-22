from jedi.inference.compiled.value import CompiledValue, CompiledName, \
from jedi.inference.base_value import LazyValueWrapper
class ExactValue(LazyValueWrapper):
    """
    This class represents exact values, that makes operations like additions
    and exact boolean values possible, while still being a "normal" stub.
    """

    def __init__(self, compiled_value):
        self.inference_state = compiled_value.inference_state
        self._compiled_value = compiled_value

    def __getattribute__(self, name):
        if name in ('get_safe_value', 'execute_operation', 'access_handle', 'negate', 'py__bool__', 'is_compiled'):
            return getattr(self._compiled_value, name)
        return super().__getattribute__(name)

    def _get_wrapped_value(self):
        instance, = builtin_from_name(self.inference_state, self._compiled_value.name.string_name).execute_with_values()
        return instance

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self._compiled_value)