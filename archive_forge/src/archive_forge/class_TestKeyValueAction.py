import argparse
from osc_lib.cli import parseractions
from osc_lib.tests import utils
class TestKeyValueAction(utils.TestCase):

    def setUp(self):
        super(TestKeyValueAction, self).setUp()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--property', metavar='<key=value>', action=parseractions.KeyValueAction, default={'green': '20%', 'format': '#rgb'}, help='Property to store for this volume (repeat option to set multiple properties)')

    def test_good_values(self):
        results = self.parser.parse_args(['--property', 'red=', '--property', 'green=100%', '--property', 'blue=50%'])
        actual = getattr(results, 'property', {})
        expect = {'red': '', 'green': '100%', 'blue': '50%', 'format': '#rgb'}
        self.assertEqual(expect, actual)

    def test_error_values(self):
        data_list = [['--property', 'red'], ['--property', '='], ['--property', '=red']]
        for data in data_list:
            self.assertRaises(argparse.ArgumentTypeError, self.parser.parse_args, data)