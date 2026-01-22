from decimal import Decimal
from tests.compat import unittest
from boto.compat import six, json
from boto.dynamodb import types
from boto.dynamodb.exceptions import DynamoDBNumberError
class TestDynamizer(unittest.TestCase):

    def setUp(self):
        pass

    def test_encoding_to_dynamodb(self):
        dynamizer = types.Dynamizer()
        self.assertEqual(dynamizer.encode('foo'), {'S': 'foo'})
        self.assertEqual(dynamizer.encode(54), {'N': '54'})
        self.assertEqual(dynamizer.encode(Decimal('1.1')), {'N': '1.1'})
        self.assertEqual(dynamizer.encode(set([1, 2, 3])), {'NS': ['1', '2', '3']})
        self.assertIn(dynamizer.encode(set(['foo', 'bar'])), ({'SS': ['foo', 'bar']}, {'SS': ['bar', 'foo']}))
        self.assertEqual(dynamizer.encode(types.Binary(b'\x01')), {'B': 'AQ=='})
        self.assertEqual(dynamizer.encode(set([types.Binary(b'\x01')])), {'BS': ['AQ==']})
        self.assertEqual(dynamizer.encode(['foo', 54, [1]]), {'L': [{'S': 'foo'}, {'N': '54'}, {'L': [{'N': '1'}]}]})
        self.assertEqual(dynamizer.encode({'foo': 'bar', 'hoge': {'sub': 1}}), {'M': {'foo': {'S': 'bar'}, 'hoge': {'M': {'sub': {'N': '1'}}}}})
        self.assertEqual(dynamizer.encode(None), {'NULL': True})
        self.assertEqual(dynamizer.encode(False), {'BOOL': False})

    def test_decoding_to_dynamodb(self):
        dynamizer = types.Dynamizer()
        self.assertEqual(dynamizer.decode({'S': 'foo'}), 'foo')
        self.assertEqual(dynamizer.decode({'N': '54'}), 54)
        self.assertEqual(dynamizer.decode({'N': '1.1'}), Decimal('1.1'))
        self.assertEqual(dynamizer.decode({'NS': ['1', '2', '3']}), set([1, 2, 3]))
        self.assertEqual(dynamizer.decode({'SS': ['foo', 'bar']}), set(['foo', 'bar']))
        self.assertEqual(dynamizer.decode({'B': 'AQ=='}), types.Binary(b'\x01'))
        self.assertEqual(dynamizer.decode({'BS': ['AQ==']}), set([types.Binary(b'\x01')]))
        self.assertEqual(dynamizer.decode({'L': [{'S': 'foo'}, {'N': '54'}, {'L': [{'N': '1'}]}]}), ['foo', 54, [1]])
        self.assertEqual(dynamizer.decode({'M': {'foo': {'S': 'bar'}, 'hoge': {'M': {'sub': {'N': '1'}}}}}), {'foo': 'bar', 'hoge': {'sub': 1}})
        self.assertEqual(dynamizer.decode({'NULL': True}), None)
        self.assertEqual(dynamizer.decode({'BOOL': False}), False)

    def test_float_conversion_errors(self):
        dynamizer = types.Dynamizer()
        self.assertEqual(dynamizer.encode(1.25), {'N': '1.25'})
        with self.assertRaises(DynamoDBNumberError):
            dynamizer.encode(1.1)

    def test_non_boolean_conversions(self):
        dynamizer = types.NonBooleanDynamizer()
        self.assertEqual(dynamizer.encode(True), {'N': '1'})

    def test_lossy_float_conversions(self):
        dynamizer = types.LossyFloatDynamizer()
        self.assertEqual(dynamizer.encode(1.1), {'N': '1.1'})
        self.assertEqual(dynamizer.decode({'N': '1.1'}), 1.1)
        self.assertEqual(dynamizer.encode(set([1.1])), {'NS': ['1.1']})
        self.assertEqual(dynamizer.decode({'NS': ['1.1', '2.2', '3.3']}), set([1.1, 2.2, 3.3]))

    def test_decoding_full_doc(self):
        """Simple List decoding that had caused some errors"""
        dynamizer = types.Dynamizer()
        doc = '{"__type__":{"S":"Story"},"company_tickers":{"SS":["NASDAQ-TSLA","NYSE-F","NYSE-GM"]},"modified_at":{"N":"1452525162"},"created_at":{"N":"1452525162"},"version":{"N":"1"},"categories":{"SS":["AUTOMTVE","LTRTR","MANUFCTU","PN","PRHYPE","TAXE","TJ","TL"]},"provider_categories":{"L":[{"S":"F"},{"S":"GM"},{"S":"TSLA"}]},"received_at":{"S":"2016-01-11T11:26:31Z"}}'
        output_doc = {'provider_categories': ['F', 'GM', 'TSLA'], '__type__': 'Story', 'company_tickers': set(['NASDAQ-TSLA', 'NYSE-GM', 'NYSE-F']), 'modified_at': Decimal('1452525162'), 'version': Decimal('1'), 'received_at': '2016-01-11T11:26:31Z', 'created_at': Decimal('1452525162'), 'categories': set(['LTRTR', 'TAXE', 'MANUFCTU', 'TL', 'TJ', 'AUTOMTVE', 'PRHYPE', 'PN'])}
        self.assertEqual(json.loads(doc, object_hook=dynamizer.decode), output_doc)