import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def try_assertion():
    with translate_exceptions:
        assert False, 'This is a test'