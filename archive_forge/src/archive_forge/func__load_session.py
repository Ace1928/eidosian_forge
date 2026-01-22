from keystoneauth1.exceptions import catalog
from keystoneauth1 import session as ksa_session
import os_client_config
from oslo_utils import importutils
from magnumclient.common import httpclient
from magnumclient.v1 import certificates
from magnumclient.v1 import cluster_templates
from magnumclient.v1 import clusters
from magnumclient.v1 import mservices
from magnumclient.v1 import nodegroups
from magnumclient.v1 import quotas
from magnumclient.v1 import stats
def _load_session(cloud=None, insecure=False, timeout=None, **kwargs):
    cloud_config = os_client_config.OpenStackConfig()
    cloud_config = cloud_config.get_one_cloud(cloud=cloud, verify=not insecure, **kwargs)
    verify, cert = cloud_config.get_requests_verify_args()
    auth = cloud_config.get_auth()
    session = ksa_session.Session(auth=auth, verify=verify, cert=cert, timeout=timeout)
    return session