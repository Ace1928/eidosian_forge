import itertools
from oslo_serialization import jsonutils
import webob
def _get_system_scope(auth_ref):
    """Return the scope information of a system scoped token."""
    if auth_ref.system_scoped:
        if auth_ref.system.get('all'):
            return 'all'