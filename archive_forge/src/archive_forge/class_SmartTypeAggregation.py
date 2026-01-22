import abc
import collections
import datetime
from dateutil import tz
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
from yaql import yaql_interface
class SmartTypeAggregation(SmartType, metaclass=abc.ABCMeta):
    __slots__ = ('types',)

    def __init__(self, *args, **kwargs):
        self.nullable = kwargs.pop('nullable', False)
        super(SmartTypeAggregation, self).__init__(self.nullable)
        self.types = []
        for item in args:
            if isinstance(item, (type, tuple)):
                item = PythonType(item)
            if isinstance(item, (HiddenParameterType, LazyParameterType)):
                raise ValueError('Special smart types are not supported')
            self.types.append(item)