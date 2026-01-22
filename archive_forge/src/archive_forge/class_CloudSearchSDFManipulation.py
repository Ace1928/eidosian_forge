from tests.unit import unittest
from httpretty import HTTPretty
from mock import MagicMock
import json
from boto.cloudsearch.document import DocumentServiceConnection
from boto.cloudsearch.document import CommitMismatchError, EncodingError, \
import boto
class CloudSearchSDFManipulation(CloudSearchDocumentTest):
    response = {'status': 'success', 'adds': 1, 'deletes': 0}

    def test_cloudsearch_initial_sdf_is_blank(self):
        document = DocumentServiceConnection(endpoint='doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')
        self.assertEqual(document.get_sdf(), '[]')

    def test_cloudsearch_single_document_sdf(self):
        document = DocumentServiceConnection(endpoint='doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')
        document.add('1234', 10, {'id': '1234', 'title': 'Title 1', 'category': ['cat_a', 'cat_b', 'cat_c']})
        self.assertNotEqual(document.get_sdf(), '[]')
        document.clear_sdf()
        self.assertEqual(document.get_sdf(), '[]')