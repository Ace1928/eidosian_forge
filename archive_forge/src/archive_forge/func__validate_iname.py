import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def _validate_iname(self, iname, state):
    """Validate an i-name"""
    iname = iname[1:]
    if '..' in iname or '--' in iname:
        raise Invalid(self.message('repeatedChar', state), iname, state)
    if self.iname_invalid_start.match(iname):
        raise Invalid(self.message('badInameStart', state), iname, state)
    if not self.iname_valid_pattern.match(iname) or '_' in iname:
        raise Invalid(self.message('badIname', state, iname=iname), iname, state)