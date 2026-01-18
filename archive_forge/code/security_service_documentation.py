from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
A resource that implements security service of Manila.

    A security_service is a set of options that defines a security domain
    for a particular shared filesystem protocol, such as an
    Active Directory domain or a Kerberos domain.
    