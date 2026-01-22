from openstackclient.api import api
def _munge_url(self):
    if not self.endpoint.endswith(self._endpoint_suffix):
        self.endpoint = self.endpoint + self._endpoint_suffix