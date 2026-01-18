import logging
from unittest import mock
from osc_lib import logs
from osc_lib.tests import utils
@mock.patch('warnings.simplefilter')
def test_set_warning_filter(self, simplefilter):
    logs.set_warning_filter(logging.ERROR)
    simplefilter.assert_called_with('ignore')
    logs.set_warning_filter(logging.WARNING)
    simplefilter.assert_called_with('ignore')
    logs.set_warning_filter(logging.INFO)
    simplefilter.assert_called_with('once')