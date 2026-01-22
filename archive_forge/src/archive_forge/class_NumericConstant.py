import abc
import collections
import datetime
from dateutil import tz
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
from yaql import yaql_interface
class NumericConstant(Constant):
    __slots__ = tuple()

    def __init__(self, nullable=False, expand=True):
        super(NumericConstant, self).__init__(nullable, expand)

    def check(self, value, context, *args, **kwargs):
        return super(NumericConstant, self).check(value, context, *args, **kwargs) and (value is None or (isinstance(value.value, (int, float)) and type(value.value) is not bool))