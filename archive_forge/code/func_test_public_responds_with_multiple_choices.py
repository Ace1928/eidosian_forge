import requests
import testtools.matchers
from keystone.tests.functional import core as functests
def test_public_responds_with_multiple_choices(self):
    resp = requests.get(self.PUBLIC_URL)
    self.assertThat(resp.status_code, is_multiple_choices)