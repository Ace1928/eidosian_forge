import sys
import glance_store
from oslo_config import cfg
from oslo_upgradecheck import common_checks
from oslo_upgradecheck import upgradecheck
from glance.common import removed_config
from glance.common import wsgi  # noqa
def _check_owner_is_tenant(self):
    if CONF.owner_is_tenant is False:
        return upgradecheck.Result(FAILURE, 'The "owner_is_tenant" option has been removed and there is no upgrade path for installations that had this option set to False.')
    return upgradecheck.Result(SUCCESS)