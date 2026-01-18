import logging
import urllib.parse
from oslo_config import cfg
from glance_store import exceptions
from glance_store.i18n import _
@staticmethod
def validate_schemas(uri, valid_schemas):
    """check if uri scheme is one of valid_schemas
        generate exception otherwise
        """
    for valid_schema in valid_schemas:
        if uri.startswith(valid_schema):
            return
    reason = _('Location URI must start with one of the following schemas: %s') % ', '.join(valid_schemas)
    LOG.warning(reason)
    raise exceptions.BadStoreUri(message=reason)