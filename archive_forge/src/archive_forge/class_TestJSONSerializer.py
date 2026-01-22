import unittest
from prov.model import ProvDocument
from prov.tests.utility import RoundTripTestCase
from prov.tests.test_model import AllTestsBase
import logging
class TestJSONSerializer(unittest.TestCase):

    def test_decoding_unicode_value(self):
        unicode_char = '’'
        json_content = '{\n    "prefix": {\n        "ex": "http://www.example.org"\n    },\n    "entity": {\n        "ex:unicode_char": {\n            "prov:label": "%s"\n        }\n    }\n}' % unicode_char
        prov_doc = ProvDocument.deserialize(content=json_content, format='json')
        e1 = prov_doc.get_record('ex:unicode_char')[0]
        self.assertIn(unicode_char, e1.get_attribute('prov:label'))