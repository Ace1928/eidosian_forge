import argparse
from osc_lib.cli import parseractions
from osc_lib.tests import utils
def test_required_keys_not_list(self):
    self.assertRaises(TypeError, self.parser.add_argument, '--test-required-dict', metavar='req1=xxx,req2=yyy', action=parseractions.MultiKeyValueAction, dest='test_required_dict', default=None, required_keys={'aaa': 'bbb'}, optional_keys=['opt1', 'opt2'], help='Test')