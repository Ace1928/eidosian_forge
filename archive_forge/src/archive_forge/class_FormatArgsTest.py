import collections
from zunclient.common import cliutils
from zunclient.common import utils
from zunclient import exceptions as exc
from zunclient.tests.unit import utils as test_utils
class FormatArgsTest(test_utils.BaseTestCase):

    def test_format_args_none(self):
        self.assertEqual({}, utils.format_args(None))

    def test_format_args(self):
        li = utils.format_args(['K1=V1,K2=V2,K3=V3,K4=V4,K5=V5'])
        self.assertEqual({'K1': 'V1', 'K2': 'V2', 'K3': 'V3', 'K4': 'V4', 'K5': 'V5'}, li)

    def test_format_args_semicolon(self):
        li = utils.format_args(['K1=V1;K2=V2;K3=V3;K4=V4;K5=V5'])
        self.assertEqual({'K1': 'V1', 'K2': 'V2', 'K3': 'V3', 'K4': 'V4', 'K5': 'V5'}, li)

    def test_format_args_mix_commas_semicolon(self):
        li = utils.format_args(['K1=V1,K2=V2,K3=V3;K4=V4,K5=V5'])
        self.assertEqual({'K1': 'V1', 'K2': 'V2', 'K3': 'V3', 'K4': 'V4', 'K5': 'V5'}, li)

    def test_format_args_split(self):
        li = utils.format_args(['K1=V1,K2=V22222222222222222222222222222222222222222222222222222222,K3=3.3.3.3'])
        self.assertEqual({'K1': 'V1', 'K2': 'V22222222222222222222222222222222222222222222222222222222', 'K3': '3.3.3.3'}, li)

    def test_format_args_multiple(self):
        li = utils.format_args(['K1=V1', 'K2=V22222222222222222222222222222222222222222222222222222222', 'K3=3.3.3.3'])
        self.assertEqual({'K1': 'V1', 'K2': 'V22222222222222222222222222222222222222222222222222222222', 'K3': '3.3.3.3'}, li)

    def test_format_args_multiple_colon_values(self):
        li = utils.format_args(['K1=V1', 'K2=V2,V22,V222,V2222', 'K3=3.3.3.3'])
        self.assertEqual({'K1': 'V1', 'K2': 'V2,V22,V222,V2222', 'K3': '3.3.3.3'}, li)

    def test_format_args_parse_comma_false(self):
        li = utils.format_args(['K1=V1,K2=2.2.2.2,K=V'], parse_comma=False)
        self.assertEqual({'K1': 'V1,K2=2.2.2.2,K=V'}, li)

    def test_format_args_multiple_values_per_args(self):
        li = utils.format_args(['K1=V1', 'K1=V2'])
        self.assertIn('K1', li)
        self.assertIn('V1', li['K1'])
        self.assertIn('V2', li['K1'])

    def test_format_args_bad_arg(self):
        args = ['K1=V1,K22.2.2.2']
        ex = self.assertRaises(exc.CommandError, utils.format_args, args)
        self.assertEqual('arguments must be a list of KEY=VALUE not K22.2.2.2', str(ex))

    def test_format_multiple_bad_args(self):
        args = ['K1=V1', 'K22.2.2.2']
        ex = self.assertRaises(exc.CommandError, utils.format_args, args)
        self.assertEqual('arguments must be a list of KEY=VALUE not K22.2.2.2', str(ex))