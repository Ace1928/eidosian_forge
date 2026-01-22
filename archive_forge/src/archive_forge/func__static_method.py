from unittest import mock
from oslotest import base as test_base
from oslo_log import helpers
@helpers.log_method_call
def _static_method():
    pass