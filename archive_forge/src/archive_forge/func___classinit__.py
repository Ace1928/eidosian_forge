import gettext
import os
import re
import textwrap
import warnings
from . import declarative
@staticmethod
def __classinit__(cls, new_attrs):
    Validator.__classinit__(cls, new_attrs)
    cls._inheritance_level += 1
    if '_deprecated_methods' in new_attrs:
        cls._deprecated_methods = cls._deprecated_methods + new_attrs['_deprecated_methods']
    for old, new in cls._deprecated_methods:
        if old in new_attrs:
            if new not in new_attrs:
                deprecation_warning(old, new, stacklevel=cls._inheritance_level + 2)
                setattr(cls, new, new_attrs[old])
        elif new in new_attrs:
            setattr(cls, old, deprecated(old=old, new=new)(new_attrs[new]))