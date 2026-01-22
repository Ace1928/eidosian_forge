import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class InsufficientAuthMethods(Error):
    message_format = _('Insufficient auth methods received for %(user_id)s. Auth Methods Provided: %(methods)s.')
    code = 401
    title = 'Unauthorized'

    def __init__(self, message=None, user_id=None, methods=None):
        methods_str = '[%s]' % ','.join(methods)
        super(InsufficientAuthMethods, self).__init__(message, user_id=user_id, methods=methods_str)
        self.user_id = user_id
        self.methods = methods