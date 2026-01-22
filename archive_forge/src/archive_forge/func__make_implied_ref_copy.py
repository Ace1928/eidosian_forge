import copy
import itertools
from oslo_log import log
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
def _make_implied_ref_copy(prior_ref, implied_role_id):
    implied_ref = copy.deepcopy(prior_ref)
    implied_ref['role_id'] = implied_role_id
    indirect = implied_ref.setdefault('indirect', {})
    indirect['role_id'] = prior_ref['role_id']
    return implied_ref