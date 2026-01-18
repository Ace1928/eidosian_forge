import pdb
import os
import ast
import pickle
import re
import time
import logging
import importlib
import tempfile
import warnings
from xml.etree import ElementTree
from elementpath.etree import PyElementTree, etree_tostring
import xmlschema
from xmlschema import XMLSchemaBase, XMLSchema11, XMLSchemaValidationError, \
from xmlschema.names import XSD_IMPORT
from xmlschema.helpers import local_name
from xmlschema.resources import fetch_namespaces
from xmlschema.validators import XsdType, Xsd11ComplexType
from xmlschema.dataobjects import DataElementConverter, DataBindingConverter, DataElement
from ._helpers import iter_nested_items, etree_elements_assert_equal
from ._case_class import XsdValidatorTestCase
from ._observers import SchemaObserver
def test_xsd_file(self):
    if inspect:
        SchemaObserver.clear()
    del self.errors[:]
    start_time = time.time()
    if expected_warnings > 0:
        with warnings.catch_warnings(record=True) as include_import_warnings:
            warnings.simplefilter('always')
            self.check_xsd_file()
            self.assertEqual(len(include_import_warnings), expected_warnings, msg=xsd_file)
    else:
        self.check_xsd_file()
    if check_with_lxml and lxml_etree is not None:
        self.check_xsd_file_with_lxml(xmlschema_time=time.time() - start_time)
    self.check_errors(xsd_file, expected_errors)