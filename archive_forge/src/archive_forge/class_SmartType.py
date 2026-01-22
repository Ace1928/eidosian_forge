import abc
import collections
import datetime
from dateutil import tz
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
from yaql import yaql_interface
class SmartType(metaclass=abc.ABCMeta):
    __slots__ = ('nullable',)

    def __init__(self, nullable):
        self.nullable = nullable

    def check(self, value, context, engine, *args, **kwargs):
        if value is None and (not self.nullable):
            return False
        return True

    def convert(self, value, receiver, context, function_spec, engine, *args, **kwargs):
        if not self.check(value, context, engine, *args, **kwargs):
            raise exceptions.ArgumentValueException()
        utils.limit_memory_usage(engine, (1, value))
        return value

    def is_specialization_of(self, other):
        return False