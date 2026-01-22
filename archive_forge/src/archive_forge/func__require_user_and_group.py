from debtcollector import renames
from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _require_user_and_group(self, user, group):
    if not (user and group):
        msg = _('Specify both a user and a group')
        raise exceptions.ValidationError(msg)