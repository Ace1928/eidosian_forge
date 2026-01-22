import abc
import collections
import datetime
from dateutil import tz
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
from yaql import yaql_interface
class NotOfType(SmartType):
    __slots__ = ('smart_type',)

    def __init__(self, smart_type, nullable=True):
        if isinstance(smart_type, (type, tuple)):
            smart_type = PythonType(smart_type, nullable=nullable)
        self.smart_type = smart_type
        super(NotOfType, self).__init__(nullable)

    def check(self, value, context, engine, *args, **kwargs):
        if isinstance(value, expressions.Constant):
            value = value.value
        if not super(NotOfType, self).check(value, context, engine, *args, **kwargs):
            return False
        if value is None or isinstance(value, expressions.Expression):
            return True
        return not self.smart_type.check(value, context, engine, *args, **kwargs)