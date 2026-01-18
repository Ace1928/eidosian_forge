from heat.common.i18n import _
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
A resource for creating manila share type.

    A share_type is an administrator-defined "type of service", comprised of
    a tenant visible description, and a list of non-tenant-visible key/value
    pairs (extra_specs) which the Manila scheduler uses to make scheduling
    decisions for shared filesystem tasks.

    Please note that share type is intended to use mostly by administrators.
    So it is very likely that Manila will prohibit creation of the resource
    without administration grants.
    