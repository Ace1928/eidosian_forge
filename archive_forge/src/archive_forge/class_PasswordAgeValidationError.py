import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class PasswordAgeValidationError(PasswordValidationError):
    message_format = _('You cannot change your password at this time due to the minimum password age. Once you change your password, it must be used for %(min_age_days)d day(s) before it can be changed. Please try again in %(days_left)d day(s) or contact your administrator to reset your password.')