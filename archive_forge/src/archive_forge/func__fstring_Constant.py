import ast
import io
import sys
import tokenize
def _fstring_Constant(self, t, write):
    assert isinstance(t.value, str)
    value = t.value.replace('{', '{{').replace('}', '}}')
    write(value)