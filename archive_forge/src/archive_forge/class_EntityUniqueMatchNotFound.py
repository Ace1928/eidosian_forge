from heat.common import exception
from heat.common.i18n import _
class EntityUniqueMatchNotFound(EntityMatchNotFound):
    msg_fmt = _('No %(entity)s unique match found for %(args)s.')