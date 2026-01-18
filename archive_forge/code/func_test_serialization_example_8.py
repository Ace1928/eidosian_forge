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
def test_serialization_example_8(self):
    """
        Test the serialization of example 8 which deals with generation.
        """
    document = prov.ProvDocument()
    document.add_namespace(*EX_NS)
    e1 = document.entity('ex:e1')
    a1 = document.activity('ex:a1')
    document.wasGeneratedBy(entity=e1, activity=a1, time='2001-10-26T21:32:52', other_attributes={'ex:port': 'p1'})
    e2 = document.entity('ex:e2')
    document.wasGeneratedBy(entity=e2, activity=a1, time='2001-10-26T10:00:00', other_attributes={'ex:port': 'p2'})
    with io.BytesIO() as actual:
        document.serialize(format='xml', destination=actual)
        compare_xml(os.path.join(DATA_PATH, 'example_08.xml'), actual)