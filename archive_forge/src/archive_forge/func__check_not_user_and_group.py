from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _check_not_user_and_group(self, user, group):
    if user and group:
        msg = _('Specify either a user or group, not both')
        raise exceptions.ValidationError(msg)