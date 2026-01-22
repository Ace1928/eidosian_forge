from oslo_log import log as logging
from urllib import parse
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
@staticmethod
def _build_meta_headers(obj_type, meta_props):
    """Returns a new dict.

        Each key of new dict is prepended with "X-Container-Meta-".
        """
    if meta_props is None:
        return {}
    return dict((('X-' + obj_type.title() + '-Meta-' + k, v) for k, v in meta_props.items()))