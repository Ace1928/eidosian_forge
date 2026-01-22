from saharaclient.api import job_types as jt
from saharaclient.tests.unit import base
class JobTypesTest(base.BaseTestCase):
    body = {'name': 'Hive', 'plugins': [{'description': 'The Apache Vanilla plugin.', 'name': 'vanilla', 'title': 'Vanilla Apache Hadoop', 'versions': {'1.2.1': {}}}, {'description': 'The Hortonworks Sahara plugin.', 'name': 'hdp', 'title': 'Hortonworks Data Platform', 'versions': {'1.3.2': {}, '2.0.6': {}}}]}

    def test_job_types_list(self):
        url = self.URL + '/job-types'
        self.responses.get(url, json={'job_types': [self.body]})
        resp = self.client.job_types.list()
        self.assertEqual(url, self.responses.last_request.url)
        self.assertIsInstance(resp[0], jt.JobType)
        self.assertFields(self.body, resp[0])