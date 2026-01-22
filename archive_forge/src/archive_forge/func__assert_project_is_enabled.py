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
def _assert_project_is_enabled(self, project_ref):
    try:
        PROVIDERS.resource_api.assert_project_enabled(project_id=project_ref['id'], project=project_ref)
    except AssertionError as e:
        LOG.warning(e)
        raise exception.Unauthorized from e