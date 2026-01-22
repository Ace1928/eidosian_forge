import http.client
from keystone.tests.unit import rest
class HealthCheckTestCase(rest.RestfulTestCase):

    def test_get_healthcheck(self):
        with self.test_client() as c:
            resp = c.get('/healthcheck', expected_status_code=http.client.OK)
            self.assertEqual(0, resp.content_length)