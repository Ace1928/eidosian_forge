import typing as ty
from openstack import exceptions
from openstack import resource
def _set_metadata(self, headers):
    self.metadata = dict()
    for header in headers:
        if header.lower().startswith(self._custom_metadata_prefix.lower()):
            key = header[len(self._custom_metadata_prefix):].lower()
            self.metadata[key] = headers[header]