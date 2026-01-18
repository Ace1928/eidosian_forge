import re
import jsonschema
from oslo_config import cfg
from oslo_log import log
from keystone import exception
from keystone.i18n import _
def validate_password(password):
    pattern = CONF.security_compliance.password_regex
    if pattern:
        if not isinstance(password, str):
            detail = _('Password must be a string type')
            raise exception.PasswordValidationError(detail=detail)
        try:
            if not re.match(pattern, password):
                pattern_desc = CONF.security_compliance.password_regex_description
                raise exception.PasswordRequirementsValidationError(detail=pattern_desc)
        except re.error:
            msg = 'Unable to validate password due to invalid regular expression - password_regex: %s'
            LOG.error(msg, pattern)
            detail = _('Unable to validate password due to invalid configuration')
            raise exception.PasswordValidationError(detail=detail)