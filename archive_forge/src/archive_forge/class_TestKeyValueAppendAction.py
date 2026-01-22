import argparse
from osc_lib.cli import parseractions
from osc_lib.tests import utils
class TestKeyValueAppendAction(utils.TestCase):

    def setUp(self):
        super(TestKeyValueAppendAction, self).setUp()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--hint', metavar='<key=value>', action=parseractions.KeyValueAppendAction, help='Arbitrary key/value pairs to be sent to the scheduler for custom use')

    def test_good_values(self):
        print(self.parser._get_optional_actions())
        results = self.parser.parse_args(['--hint', 'same_host=a0cf03a5-d921-4877-bb5c-86d26cf818e1', '--hint', 'same_host=8c19174f-4220-44f0-824a-cd1eeef10287', '--hint', 'query=[>=,$free_ram_mb,1024]'])
        actual = getattr(results, 'hint', {})
        expect = {'same_host': ['a0cf03a5-d921-4877-bb5c-86d26cf818e1', '8c19174f-4220-44f0-824a-cd1eeef10287'], 'query': ['[>=,$free_ram_mb,1024]']}
        self.assertEqual(expect, actual)

    def test_error_values(self):
        data_list = [['--hint', 'red'], ['--hint', '='], ['--hint', '=red']]
        for data in data_list:
            self.assertRaises(argparse.ArgumentTypeError, self.parser.parse_args, data)