from heat.common import exception
from heat.common.i18n import _
class EntityMatchNotFound(exception.HeatException):
    msg_fmt = _('No %(entity)s matching %(args)s.')

    def __init__(self, entity=None, args=None, **kwargs):
        super(EntityMatchNotFound, self).__init__(entity=entity, args=args, **kwargs)