import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class ImmutableParameterModified(HeatException):
    msg_fmt = _('The following parameters are immutable and may not be updated: %(keys)s')

    def __init__(self, *args, **kwargs):
        if args:
            kwargs.update({'keys': ', '.join(args)})
        super(ImmutableParameterModified, self).__init__(**kwargs)