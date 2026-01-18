import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
@excutils.exception_filter
def translate_exceptions(ex):
    raise RuntimeError