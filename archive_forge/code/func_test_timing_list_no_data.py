import datetime
from keystoneauth1 import session
from osc_lib.command import timing
from osc_lib.tests import fakes
from osc_lib.tests import utils
def test_timing_list_no_data(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    datalist = [('Total', 0.0)]
    self.assertEqual(datalist, data)