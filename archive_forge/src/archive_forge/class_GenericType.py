import abc
import collections
import datetime
from dateutil import tz
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
from yaql import yaql_interface
class GenericType(SmartType):
    __slots__ = ('checker', 'converter')

    def __init__(self, nullable, checker=None, converter=None):
        super(GenericType, self).__init__(nullable)
        self.checker = checker
        self.converter = converter

    def check(self, value, context, engine, *args, **kwargs):
        if isinstance(value, expressions.Constant):
            value = value.value
        if not super(GenericType, self).check(value, context, engine, *args, **kwargs):
            return False
        if value is None or isinstance(value, expressions.Expression):
            return True
        if not self.checker:
            return True
        return self.checker(value, context, *args, **kwargs)

    def convert(self, value, receiver, context, function_spec, engine, *args, **kwargs):
        if isinstance(value, expressions.Constant):
            value = value.value
        super(GenericType, self).convert(value, receiver, context, function_spec, engine, *args, **kwargs)
        if value is None or not self.converter:
            return value
        return self.converter(value, receiver, context, function_spec, engine, *args, **kwargs)