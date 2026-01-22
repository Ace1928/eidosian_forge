import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def _validateReturn(self, field_dict, state):
    ccType = str(field_dict[self.cc_type_field]).strip()
    ccCode = str(field_dict[self.cc_code_field]).strip()
    try:
        int(ccCode)
    except ValueError:
        return {self.cc_code_field: self.message('notANumber', state)}
    length = self._cardInfo[ccType]
    if len(ccCode) != length:
        return {self.cc_code_field: self.message('badLength', state)}