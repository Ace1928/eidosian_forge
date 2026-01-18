import argparse
from unittest import mock
import testtools
from aodhclient.v2 import alarm_cli
def test_validate_time_constraint(self):
    starts = ['0 11 * * *', ' 0 11 * * * ', '"0 11 * * *"', "'0 11 * * *'"]
    for start in starts:
        string = 'name=const1;start=%s;duration=1' % start
        expected = dict(name='const1', start='0 11 * * *', duration='1')
        self.assertEqual(expected, self.cli_alarm_create.validate_time_constraint(string))