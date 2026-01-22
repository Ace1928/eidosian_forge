import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def _make_filter_classmethod(self, ignore_classes=AssertionError):

    class ExceptionIgnorer(object):
        ignore = ignore_classes

        @excutils.exception_filter
        @classmethod
        def ignore_exceptions(cls, ex):
            """Ignore some exceptions C."""
            return isinstance(ex, cls.ignore)
    return ExceptionIgnorer.ignore_exceptions