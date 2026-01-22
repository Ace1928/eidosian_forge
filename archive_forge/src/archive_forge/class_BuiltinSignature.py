from inspect import Parameter
from jedi.cache import memoize_method
from jedi import debug
from jedi import parser_utils
class BuiltinSignature(AbstractSignature):

    def __init__(self, value, return_string, function_value=None, is_bound=False):
        super().__init__(value, is_bound)
        self._return_string = return_string
        self.__function_value = function_value

    @property
    def annotation_string(self):
        return self._return_string

    @property
    def _function_value(self):
        if self.__function_value is None:
            return self.value
        return self.__function_value

    def bind(self, value):
        return BuiltinSignature(value, self._return_string, function_value=self.value, is_bound=True)