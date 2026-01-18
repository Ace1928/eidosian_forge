from oslo_log import log as logging
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
Convert a dict to a frozenset.

            Create an immutable equivalent of a dict, so it's hashable
            therefore can be used as an element of a set or a key of another
            dictionary.
            