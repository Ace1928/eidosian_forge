from keystoneauth1 import access
from keystoneauth1.identity import v3
from keystoneauth1.identity.v3 import federation
def _requests_auth(mutual_authentication):
    return requests_kerberos.HTTPKerberosAuth(mutual_authentication=_mutual_auth(mutual_authentication))