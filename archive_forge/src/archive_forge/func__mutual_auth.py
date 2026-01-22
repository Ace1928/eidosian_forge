from keystoneauth1 import access
from keystoneauth1.identity import v3
from keystoneauth1.identity.v3 import federation
def _mutual_auth(value):
    if value is None:
        return requests_kerberos.OPTIONAL
    return {'required': requests_kerberos.REQUIRED, 'optional': requests_kerberos.OPTIONAL, 'disabled': requests_kerberos.DISABLED}.get(value.lower(), requests_kerberos.OPTIONAL)