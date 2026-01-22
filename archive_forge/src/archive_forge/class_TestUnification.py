import unittest
import logging
import os
from prov.model import ProvDocument, ProvBundle, ProvException, first, Literal
from prov.tests import examples
from prov.tests.attributes import TestAttributesBase
from prov.tests.qnames import TestQualifiedNamesBase
from prov.tests.statements import TestStatementsBase
from prov.tests.utility import RoundTripTestCase
class TestUnification(unittest.TestCase):

    def test_unifying(self):
        json_path = os.path.dirname(os.path.abspath(__file__)) + '/unification/'
        filenames = os.listdir(json_path)
        for filename in filenames:
            if not filename.endswith('.json'):
                continue
            filepath = json_path + filename
            with open(filepath) as json_file:
                logger.info('Testing unifying: %s', filename)
                logger.debug('Loading %s...', filepath)
                document = ProvDocument.deserialize(json_file)
                flattened = document.flattened()
                unified = flattened.unified()
                self.assertLess(len(unified.get_records()), len(flattened.get_records()))