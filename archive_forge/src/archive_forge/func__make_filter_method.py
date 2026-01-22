import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def _make_filter_method(self, ignore_classes=AssertionError):

    class ExceptionIgnorer(object):

        def __init__(self, ignore):
            self.ignore = ignore

        @excutils.exception_filter
        def ignore_exceptions(self, ex):
            """Ignore some exceptions M."""
            return isinstance(ex, self.ignore)
    return ExceptionIgnorer(ignore_classes).ignore_exceptions