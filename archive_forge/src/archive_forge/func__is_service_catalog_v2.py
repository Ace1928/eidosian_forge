import logging
import keystoneauth1.access.service_catalog as sc
import keystoneauth1.identity.generic as auth_plugin
from keystoneauth1 import session as ks_session
import mistralclient.api.httpclient as api
from mistralclient import auth as mistral_auth
from oslo_serialization import jsonutils
@staticmethod
def _is_service_catalog_v2(catalog):
    """Check if the service catalog is of type ServiceCatalogV2

        :param catalog: the service catalog
        :return: True if V2, False otherwise
        """
    return type(catalog) is sc.ServiceCatalogV2