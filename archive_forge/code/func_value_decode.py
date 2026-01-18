import re
import string
import types
def value_decode(self, val):
    return (_unquote(val), val)