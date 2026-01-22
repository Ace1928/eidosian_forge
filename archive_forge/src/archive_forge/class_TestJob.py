from tests.unit import unittest
from mock import call, Mock, patch, sentinel
import codecs
from boto.glacier.layer1 import Layer1
from boto.glacier.layer2 import Layer2
import boto.glacier.vault
from boto.glacier.vault import Vault
from boto.glacier.vault import Job
from datetime import datetime, tzinfo, timedelta
class TestJob(GlacierLayer2Base):

    def setUp(self):
        GlacierLayer2Base.setUp(self)
        self.vault = Vault(self.mock_layer1, FIXTURE_VAULT)
        self.job = Job(self.vault, FIXTURE_ARCHIVE_JOB)

    def test_get_job_output(self):
        self.mock_layer1.get_job_output.return_value = 'TEST_OUTPUT'
        self.job.get_output((0, 100))
        self.mock_layer1.get_job_output.assert_called_with('examplevault', 'HkF9p6o7yjhFx-K3CGl6fuSm6VzW9T7esGQfco8nUXVYwS0jlb5gq1JZ55yHgt5vP54ZShjoQzQVVh7vEXAMPLEjobID', (0, 100))