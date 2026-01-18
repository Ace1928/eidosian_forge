import uuid
import fixtures
from keystoneauth1.fixture import v2
from keystoneauth1.fixture import v3
import os_service_types
def use_suburl(self):
    self._endpoint_templates = _SUBURL_TEMPLATES