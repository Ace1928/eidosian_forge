import inspect
import warnings
from asgiref.sync import iscoroutinefunction, markcoroutinefunction, sync_to_async
class DeprecationInstanceCheck(type):

    def __instancecheck__(self, instance):
        warnings.warn('`%s` is deprecated, use `%s` instead.' % (self.__name__, self.alternative), self.deprecation_warning, 2)
        return super().__instancecheck__(instance)