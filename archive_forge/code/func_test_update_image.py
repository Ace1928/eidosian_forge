from saharaclient.api import images
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_update_image(self):
    url = self.URL + '/images/id'
    self.responses.post(url, json={'image': self.body}, status_code=202)
    self.client.images.update_image('id', 'name', 'descr')
    self.assertEqual(url, self.responses.last_request.url)
    self.assertEqual(self.body, json.loads(self.responses.last_request.body))