from saharaclient.api import images
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_unregister_image(self):
    url = self.URL + '/images/id'
    self.responses.delete(url, status_code=204)
    self.client.images.unregister_image('id')
    self.assertEqual(url, self.responses.last_request.url)