from keystoneauth1 import access
from keystoneauth1.identity import v3
from keystoneauth1.identity.v3 import federation
class Kerberos(v3.AuthConstructor):
    _auth_method_class = KerberosMethod