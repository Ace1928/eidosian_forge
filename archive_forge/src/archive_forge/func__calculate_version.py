import warnings
from debtcollector import removals
from keystoneauth1 import plugin
from keystoneclient import _discover
from keystoneclient import exceptions
from keystoneclient.i18n import _
from keystoneclient import session as client_session
from keystoneclient.v2_0 import client as v2_client
from keystoneclient.v3 import client as v3_client
def _calculate_version(self, version, unstable):
    version_data = None
    if version:
        version_data = self.data_for(version)
    else:
        all_versions = self.version_data(unstable=unstable)
        if all_versions:
            version_data = all_versions[-1]
    if not version_data:
        msg = _('Could not find a suitable endpoint')
        if version:
            msg = _('Could not find a suitable endpoint for client version: %s') % str(version)
        raise exceptions.VersionNotAvailable(msg)
    return version_data