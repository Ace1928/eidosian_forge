from boto.compat import StringIO
from tests.compat import mock, unittest
from boto.glacier import vault
from boto.glacier.job import Job
from boto.glacier.response import GlacierResponse
def test_retrieve_inventory(self):

    class FakeResponse(object):
        status = 202

        def getheader(self, key, default=None):
            if key == 'x-amz-job-id':
                return 'HkF9p6'
            elif key == 'Content-Type':
                return 'application/json'
            return 'something'

        def read(self, amt=None):
            return b'{\n  "Action": "ArchiveRetrieval",\n  "ArchiveId": "NkbByEejwEggmBz2fTHgJrg0XBoDfjP4q6iu87-EXAMPLEArchiveId",\n  "ArchiveSizeInBytes": 16777216,\n  "ArchiveSHA256TreeHash": "beb0fe31a1c7ca8c6c04d574ea906e3f97",\n  "Completed": false,\n  "CreationDate": "2012-05-15T17:21:39.339Z",\n  "CompletionDate": "2012-05-15T17:21:43.561Z",\n  "InventorySizeInBytes": null,\n  "JobDescription": "My ArchiveRetrieval Job",\n  "JobId": "HkF9p6",\n  "RetrievalByteRange": "0-16777215",\n  "SHA256TreeHash": "beb0fe31a1c7ca8c6c04d574ea906e3f97b31fd",\n  "SNSTopic": "arn:aws:sns:us-east-1:012345678901:mytopic",\n  "StatusCode": "InProgress",\n  "StatusMessage": "Operation in progress.",\n  "VaultARN": "arn:aws:glacier:us-east-1:012345678901:vaults/examplevault"\n}'
    raw_resp = FakeResponse()
    init_resp = GlacierResponse(raw_resp, [('x-amz-job-id', 'JobId')])
    raw_resp_2 = FakeResponse()
    desc_resp = GlacierResponse(raw_resp_2, [])
    with mock.patch.object(self.vault.layer1, 'initiate_job', return_value=init_resp):
        with mock.patch.object(self.vault.layer1, 'describe_job', return_value=desc_resp):
            self.assertEqual(self.vault.retrieve_inventory(), 'HkF9p6')
            job = self.vault.retrieve_inventory_job()
            self.assertTrue(isinstance(job, Job))
            self.assertEqual(job.id, 'HkF9p6')