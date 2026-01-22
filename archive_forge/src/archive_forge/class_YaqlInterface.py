import abc
import collections
import datetime
from dateutil import tz
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
from yaql import yaql_interface
class YaqlInterface(HiddenParameterType, SmartType):
    __slots__ = tuple()

    def __init__(self):
        super(YaqlInterface, self).__init__(False)

    def convert(self, value, receiver, context, function_spec, engine, *args, **kwargs):
        return yaql_interface.YaqlInterface(context, engine, receiver)