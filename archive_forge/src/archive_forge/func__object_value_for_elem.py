import re
from .types import _StringType
from ... import exc
from ... import sql
from ... import util
from ...sql import sqltypes
def _object_value_for_elem(self, elem):
    if elem == '':
        return elem
    else:
        return super()._object_value_for_elem(elem)