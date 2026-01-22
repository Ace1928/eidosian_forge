from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.octavia import octavia_base
from heat.engine import support
from heat.engine import translation
def _prepare_session_persistence(self, props):
    session_p = props.get(self.SESSION_PERSISTENCE)
    if session_p is not None:
        session_props = dict(((k, v) for k, v in session_p.items() if v is not None))
        props[self.SESSION_PERSISTENCE] = session_props