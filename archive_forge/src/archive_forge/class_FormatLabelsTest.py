import builtins
import collections
from unittest import mock
from oslo_serialization import jsonutils
import tempfile
from magnumclient.common import cliutils
from magnumclient.common import utils
from magnumclient import exceptions as exc
from magnumclient.tests import utils as test_utils
class FormatLabelsTest(test_utils.BaseTestCase):

    def test_format_label_none(self):
        self.assertEqual({}, utils.format_labels(None))

    def test_format_labels(self):
        la = utils.format_labels(['K1=V1,K2=V2,K3=V3,K4=V4,K5=V5'])
        self.assertEqual({'K1': 'V1', 'K2': 'V2', 'K3': 'V3', 'K4': 'V4', 'K5': 'V5'}, la)

    def test_format_labels_semicolon(self):
        la = utils.format_labels(['K1=V1;K2=V2;K3=V3;K4=V4;K5=V5'])
        self.assertEqual({'K1': 'V1', 'K2': 'V2', 'K3': 'V3', 'K4': 'V4', 'K5': 'V5'}, la)

    def test_format_labels_mix_commas_semicolon(self):
        la = utils.format_labels(['K1=V1,K2=V2,K3=V3;K4=V4,K5=V5'])
        self.assertEqual({'K1': 'V1', 'K2': 'V2', 'K3': 'V3', 'K4': 'V4', 'K5': 'V5'}, la)

    def test_format_labels_split(self):
        la = utils.format_labels(['K1=V1,K2=V22222222222222222222222222222222222222222222222222222222,K3=3.3.3.3'])
        self.assertEqual({'K1': 'V1', 'K2': 'V22222222222222222222222222222222222222222222222222222222', 'K3': '3.3.3.3'}, la)

    def test_format_labels_multiple(self):
        la = utils.format_labels(['K1=V1', 'K2=V22222222222222222222222222222222222222222222222222222222', 'K3=3.3.3.3'])
        self.assertEqual({'K1': 'V1', 'K2': 'V22222222222222222222222222222222222222222222222222222222', 'K3': '3.3.3.3'}, la)

    def test_format_labels_multiple_colon_values(self):
        la = utils.format_labels(['K1=V1', 'K2=V2,V22,V222,V2222', 'K3=3.3.3.3'])
        self.assertEqual({'K1': 'V1', 'K2': 'V2,V22,V222,V2222', 'K3': '3.3.3.3'}, la)

    def test_format_labels_parse_comma_false(self):
        la = utils.format_labels(['K1=V1,K2=2.2.2.2,K=V'], parse_comma=False)
        self.assertEqual({'K1': 'V1,K2=2.2.2.2,K=V'}, la)

    def test_format_labels_multiple_values_per_labels(self):
        la = utils.format_labels(['K1=V1', 'K1=V2'])
        self.assertEqual({'K1': 'V1,V2'}, la)

    def test_format_label_special_label(self):
        labels = ['K1=V1,K22.2.2.2']
        la = utils.format_labels(labels, parse_comma=True)
        self.assertEqual({'K1': 'V1,K22.2.2.2'}, la)

    def test_format_multiple_bad_label(self):
        labels = ['K1=V1', 'K22.2.2.2']
        ex = self.assertRaises(exc.CommandError, utils.format_labels, labels)
        self.assertEqual('labels must be a list of KEY=VALUE not K22.2.2.2', str(ex))