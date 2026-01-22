from jedi import debug
from jedi.inference.base_value import ValueSet, NO_VALUES, ValueWrapper
from jedi.inference.gradual.base import BaseTypingValue
class TypeVarClass(ValueWrapper):

    def py__call__(self, arguments):
        unpacked = arguments.unpack()
        key, lazy_value = next(unpacked, (None, None))
        var_name = self._find_string_name(lazy_value)
        if var_name is None or key is not None:
            debug.warning('Found a variable without a name %s', arguments)
            return NO_VALUES
        return ValueSet([TypeVar.create_cached(self.inference_state, self.parent_context, tree_name=self.tree_node.name, var_name=var_name, unpacked_args=unpacked)])

    def _find_string_name(self, lazy_value):
        if lazy_value is None:
            return None
        value_set = lazy_value.infer()
        if not value_set:
            return None
        if len(value_set) > 1:
            debug.warning('Found multiple values for a type variable: %s', value_set)
        name_value = next(iter(value_set))
        try:
            method = name_value.get_safe_value
        except AttributeError:
            return None
        else:
            safe_value = method(default=None)
            if isinstance(safe_value, str):
                return safe_value
            return None