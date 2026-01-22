import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def _make_filter_func(self, ignore_classes=AssertionError):

    @excutils.exception_filter
    def ignore_exceptions(ex):
        """Ignore some exceptions F."""
        return isinstance(ex, ignore_classes)
    return ignore_exceptions