import webob
from heat.common import noauth
from heat.tests import common
def _start_fake_response(self, status, headers):
    self.response_status = int(status.split(' ', 1)[0])
    self.response_headers = dict(headers)