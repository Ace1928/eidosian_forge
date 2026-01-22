from collections import namedtuple
from inspect import isclass
import re
import warnings
from peewee import *
from peewee import _StringField
from peewee import _query_val_transform
from peewee import CommaNodeList
from peewee import SCOPE_VALUES
from peewee import make_snake_case
from peewee import text_type
def _clean_default(self, field_class, default):
    if default is None or field_class in (AutoField, BigAutoField) or default.lower() == 'null':
        return
    if issubclass(field_class, _StringField) and isinstance(default, text_type) and (not default.startswith("'")):
        default = "'%s'" % default
    return default or "''"