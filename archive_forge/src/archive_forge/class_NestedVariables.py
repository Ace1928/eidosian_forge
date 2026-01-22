from .api import FancyValidator
class NestedVariables(FancyValidator):

    def _convert_to_python(self, value, state):
        return variable_decode(value)

    def _convert_from_python(self, value, state):
        return variable_encode(value)

    def empty_value(self, value):
        return {}