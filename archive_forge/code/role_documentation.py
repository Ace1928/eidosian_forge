from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import translation
Heat Template Resource for Keystone Role.

    Roles dictate the level of authorization the end user can obtain. Roles can
    be granted at either the domain or project level. Role can be assigned to
    the individual user or at the group level. Role name is unique within the
    owning domain.
    