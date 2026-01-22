import types
from ._impl import (
def _str_preprocessor(self):
    if isinstance(self.preprocessor, types.FunctionType):
        return '<function %s>' % self.preprocessor.__name__
    return str(self.preprocessor)