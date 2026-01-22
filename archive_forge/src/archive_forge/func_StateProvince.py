import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def StateProvince(*kw, **kwargs):
    deprecation_warning('please use formencode.national.USStateProvince')
    from formencode.national import USStateProvince
    return USStateProvince(*kw, **kwargs)