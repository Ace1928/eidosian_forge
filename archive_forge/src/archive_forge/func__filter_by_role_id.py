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
def _filter_by_role_id(self, role_id, ref_results):
    filter_results = []
    for ref in ref_results:
        if ref['role_id'] == role_id:
            filter_results.append(ref)
    return filter_results