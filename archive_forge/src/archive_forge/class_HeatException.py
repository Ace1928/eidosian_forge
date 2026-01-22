import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class HeatException(Exception):
    """Base Heat Exception.

    To correctly use this class, inherit from it and define a 'msg_fmt'
    property. That msg_fmt will get formatted with the keyword arguments
    provided to the constructor.
    """
    message = _('An unknown exception occurred.')
    error_code = None
    safe = True

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        if self.error_code in ERROR_CODE_MAP:
            self.msg_fmt = ERROR_CODE_MAP[self.error_code]
        try:
            self.message = self.msg_fmt % kwargs
        except KeyError:
            with excutils.save_and_reraise_exception(reraise=_FATAL_EXCEPTION_FORMAT_ERRORS):
                LOG.exception('Exception in string format operation')
                for name, value in kwargs.items():
                    LOG.error('%(name)s: %(value)s', {'name': name, 'value': value})
        if self.error_code:
            self.message = 'HEAT-E%s %s' % (self.error_code, self.message)

    def __str__(self):
        return self.message

    def __deepcopy__(self, memo):
        return self.__class__(**self.kwargs)