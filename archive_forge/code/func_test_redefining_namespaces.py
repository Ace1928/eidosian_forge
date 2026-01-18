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
def test_redefining_namespaces(self):
    """
        Test the behaviour when namespaces are redefined at the element level.
        """
    filename = os.path.join(DATA_PATH, 'namespace_redefined_but_does_not_change.xml')
    doc = prov.ProvDocument.deserialize(source=filename, format='xml')
    self.assertEqual(len(doc._records), 1)
    ns = Namespace('ex', 'http://example.com/ns/ex#')
    self.assertEqual(doc._records[0].attributes[0][1].namespace, ns)
    filename = os.path.join(DATA_PATH, 'namespace_redefined.xml')
    doc = prov.ProvDocument.deserialize(source=filename, format='xml')
    new_ns = doc._records[0].attributes[0][1].namespace
    self.assertNotEqual(new_ns, ns)
    self.assertEqual(new_ns.uri, 'http://example.com/ns/new_ex#')