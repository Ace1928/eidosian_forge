import logging
import sys
from unittest import mock
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_context import context
from oslotest import base as test_base
from oslo_log import formatters
from oslo_log import log
def _unhashable_exception_info(self):

    class UnhashableException(Exception):
        __hash__ = None
    try:
        raise UnhashableException()
    except UnhashableException:
        return sys.exc_info()