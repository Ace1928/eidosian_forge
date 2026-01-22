import logging
import keystoneauth1.access.service_catalog as sc
import keystoneauth1.identity.generic as auth_plugin
from keystoneauth1 import session as ks_session
import mistralclient.api.httpclient as api
from mistralclient import auth as mistral_auth
from oslo_serialization import jsonutils
@staticmethod
def _verification_needed(cacert, insecure):
    """Return the verify parameter.

        The value of verify can be either True/False or a cacert.

        :param cacert None or path to CA cert file
        :param insecure truthy value to switch on SSL verification
        """
    if insecure is False or insecure is None:
        verify = cacert or True
    else:
        verify = False
    return verify