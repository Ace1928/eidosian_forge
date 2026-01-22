from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
Heat Template Resource for Monasca Alarm definition.

    Monasca Alarm definition helps to define the required expression for
    a given alarm situation. This plugin helps to create, update and
    delete the alarm definition.

    Alarm definitions is necessary to describe and manage alarms in a
    one-to-many relationship in order to avoid having to manually declare each
    alarm even though they may share many common attributes and differ in only
    one, such as hostname.
    