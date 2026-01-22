import json
import copy
import tempfile
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.glacier.layer1 import Layer1
from boto.compat import six
class GlacierJobOperations(GlacierLayer1ConnectionBase):

    def setUp(self):
        super(GlacierJobOperations, self).setUp()
        self.job_content = 'abc' * 1024

    def test_initiate_archive_job(self):
        content = {u'Type': u'archive-retrieval', u'ArchiveId': u'AAABZpJrTyioDC_HsOmHae8EZp_uBSJr6cnGOLKp_XJCl-Q', u'Description': u'Test Archive', u'SNSTopic': u'Topic', u'JobId': None, u'Location': None, u'RequestId': None}
        self.set_http_response(status_code=202, header=self.json_header, body=json.dumps(content).encode('utf-8'))
        api_response = self.service_connection.initiate_job(self.vault_name, self.job_content)
        self.assertDictEqual(content, api_response)

    def test_get_archive_output(self):
        header = [('Content-Type', 'application/octet-stream')]
        self.set_http_response(status_code=200, header=header, body=self.job_content)
        response = self.service_connection.get_job_output(self.vault_name, 'example-job-id')
        self.assertEqual(self.job_content, response.read())