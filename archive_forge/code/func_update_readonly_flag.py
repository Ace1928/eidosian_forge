from cinderclient.apiclient import base as common_base
from cinderclient import base
def update_readonly_flag(self, volume, flag):
    return self._action('os-update_readonly_flag', base.getid(volume), {'readonly': flag})