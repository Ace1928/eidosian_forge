import datetime
from keystoneauth1 import session
from osc_lib.command import timing
from osc_lib.tests import fakes
from osc_lib.tests import utils
def test_timing_list(self):
    self.app.timing_data = [session.RequestTiming(method=timing_method, url=timing_url, elapsed=datetime.timedelta(microseconds=timing_elapsed * 1000000))]
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    datalist = [(timing_method + ' ' + timing_url, timing_elapsed), ('Total', timing_elapsed)]
    self.assertEqual(datalist, data)