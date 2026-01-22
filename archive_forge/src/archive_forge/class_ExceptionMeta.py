import functools
import inspect
import logging
from oslo_config import cfg
from oslo_log._i18n import _
class ExceptionMeta(type):

    def __subclasscheck__(self, subclass):
        if self in _DEPRECATED_EXCEPTIONS:
            report_deprecated()
        return super(ExceptionMeta, self).__subclasscheck__(subclass)