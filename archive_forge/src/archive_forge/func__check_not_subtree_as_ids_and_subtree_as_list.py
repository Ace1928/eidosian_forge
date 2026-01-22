import urllib.parse
from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _check_not_subtree_as_ids_and_subtree_as_list(self, subtree_as_ids, subtree_as_list):
    if subtree_as_ids and subtree_as_list:
        msg = _('Specify either subtree_as_ids or subtree_as_list parameters, not both')
        raise exceptions.ValidationError(msg)