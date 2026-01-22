from keystoneclient import base
def _build_url_and_put(self, **kwargs):
    url = self.build_url(dict_args_in_out=kwargs)
    body = {self.key: kwargs}
    return self._update(url, body=body, response_key=self.key, method='PUT')