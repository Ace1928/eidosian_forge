from tests.unit import unittest
from httpretty import HTTPretty
from mock import MagicMock
import json
from boto.cloudsearch.document import DocumentServiceConnection
from boto.cloudsearch.document import CommitMismatchError, EncodingError, \
import boto
class CloudSearchDocumentSingleTest(CloudSearchDocumentTest):
    response = {'status': 'success', 'adds': 1, 'deletes': 0}

    def test_cloudsearch_add_basics(self):
        """
        Check that a simple add document actually sends an add document request
        to AWS.
        """
        document = DocumentServiceConnection(endpoint='doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')
        document.add('1234', 10, {'id': '1234', 'title': 'Title 1', 'category': ['cat_a', 'cat_b', 'cat_c']})
        document.commit()
        args = json.loads(HTTPretty.last_request.body.decode('utf-8'))[0]
        self.assertEqual(args['lang'], 'en')
        self.assertEqual(args['type'], 'add')

    def test_cloudsearch_add_single_basic(self):
        """
        Check that a simple add document sends correct document metadata to
        AWS.
        """
        document = DocumentServiceConnection(endpoint='doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')
        document.add('1234', 10, {'id': '1234', 'title': 'Title 1', 'category': ['cat_a', 'cat_b', 'cat_c']})
        document.commit()
        args = json.loads(HTTPretty.last_request.body.decode('utf-8'))[0]
        self.assertEqual(args['id'], '1234')
        self.assertEqual(args['version'], 10)
        self.assertEqual(args['type'], 'add')

    def test_cloudsearch_add_single_fields(self):
        """
        Check that a simple add document sends the actual document to AWS.
        """
        document = DocumentServiceConnection(endpoint='doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')
        document.add('1234', 10, {'id': '1234', 'title': 'Title 1', 'category': ['cat_a', 'cat_b', 'cat_c']})
        document.commit()
        args = json.loads(HTTPretty.last_request.body.decode('utf-8'))[0]
        self.assertEqual(args['fields']['category'], ['cat_a', 'cat_b', 'cat_c'])
        self.assertEqual(args['fields']['id'], '1234')
        self.assertEqual(args['fields']['title'], 'Title 1')

    def test_cloudsearch_add_single_result(self):
        """
        Check that the reply from adding a single document is correctly parsed.
        """
        document = DocumentServiceConnection(endpoint='doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')
        document.add('1234', 10, {'id': '1234', 'title': 'Title 1', 'category': ['cat_a', 'cat_b', 'cat_c']})
        doc = document.commit()
        self.assertEqual(doc.status, 'success')
        self.assertEqual(doc.adds, 1)
        self.assertEqual(doc.deletes, 0)
        self.assertEqual(doc.doc_service, document)