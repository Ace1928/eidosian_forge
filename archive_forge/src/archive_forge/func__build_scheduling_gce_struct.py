import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def _build_scheduling_gce_struct(self, on_host_maintenance=None, automatic_restart=None, preemptible=None):
    """
        Build the scheduling dict suitable for use with the GCE API.

        :param    on_host_maintenance: Defines whether node should be
                                          terminated or migrated when host
                                          machine goes down. Acceptable values
                                          are: 'MIGRATE' or 'TERMINATE' (If
                                          not supplied, value will be reset to
                                          GCE default value for the instance
                                          type.)
        :type     on_host_maintenance: ``str`` or ``None``

        :param    automatic_restart: Defines whether the instance should be
                                        automatically restarted when it is
                                        terminated by Compute Engine. (If not
                                        supplied, value will be set to the GCE
                                        default value for the instance type.)
        :type     automatic_restart: ``bool`` or ``None``

        :param    preemptible: Defines whether the instance is preemptible.
                                        (If not supplied, the instance will
                                         not be preemptible)
        :type     preemptible: ``bool`` or ``None``

        :return:  A dictionary of scheduling options for the GCE API.
        :rtype:   ``dict``
        """
    scheduling = {}
    if preemptible is not None:
        if isinstance(preemptible, bool):
            scheduling['preemptible'] = preemptible
        else:
            raise ValueError('boolean expected for preemptible')
    if on_host_maintenance is not None:
        maint_opts = ['MIGRATE', 'TERMINATE']
        if isinstance(on_host_maintenance, str) and on_host_maintenance in maint_opts:
            if preemptible is True and on_host_maintenance == 'MIGRATE':
                raise ValueError("host maintenance cannot be 'MIGRATE' if instance is preemptible.")
            scheduling['onHostMaintenance'] = on_host_maintenance
        else:
            raise ValueError('host maintenance must be one of %s' % ','.join(maint_opts))
    if automatic_restart is not None:
        if isinstance(automatic_restart, bool):
            if automatic_restart is True and preemptible is True:
                raise ValueError('instance cannot be restarted if it is preemptible.')
            scheduling['automaticRestart'] = automatic_restart
        else:
            raise ValueError('boolean expected for automatic')
    return scheduling