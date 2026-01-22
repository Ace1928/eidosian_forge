from functools import partial
from oslo_log import log
import stevedore
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.backends import resource_options as ro
def _set_scope_from_app_cred(self, app_cred_info):
    app_cred_ref = self._lookup_app_cred(app_cred_info)
    self._scope_data = (None, app_cred_ref['project_id'], None, None, None)
    return