import difflib
import glob
import inspect
import io
from lxml import etree
import os
import unittest
import warnings
from prov.identifier import Namespace, QualifiedName
from prov.constants import PROV
import prov.model as prov
from prov.tests.test_model import AllTestsBase
from prov.tests.utility import RoundTripTestCase
class ProvXMLRoundTripFromFileTestCase(unittest.TestCase):

    def _perform_round_trip(self, filename, force_types=False):
        document = prov.ProvDocument.deserialize(source=filename, format='xml')
        with io.BytesIO() as new_xml:
            document.serialize(format='xml', destination=new_xml, force_types=force_types)
            compare_xml(filename, new_xml)