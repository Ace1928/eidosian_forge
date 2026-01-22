import abc
import base64
import hashlib
import os
import time
from urllib import parse as urlparse
import warnings
from keystoneauth1 import _utils as utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import federation
def _get_discovery_document(self, session):
    """Get the contents of the OpenID Connect Discovery Document.

        This method grabs the contents of the OpenID Connect Discovery Document
        if a discovery_endpoint was passed to the constructor and returns it as
        a dict, otherwise returns an empty dict. Note that it will fetch the
        discovery document only once, so subsequent calls to this method will
        return the cached result, if any.

        :param session: a session object to send out HTTP requests.
        :type session: keystoneauth1.session.Session

        :returns: a python dictionary containing the discovery document if any,
                  otherwise it will return an empty dict.
        :rtype: dict
        """
    if self.discovery_endpoint is not None and (not self._discovery_document):
        try:
            resp = session.get(self.discovery_endpoint, authenticated=False)
        except exceptions.HttpError:
            _logger.error('Cannot fetch discovery document %(discovery)s' % {'discovery': self.discovery_endpoint})
            raise
        try:
            self._discovery_document = resp.json()
        except Exception:
            pass
        if not self._discovery_document:
            raise exceptions.InvalidOidcDiscoveryDocument()
    return self._discovery_document