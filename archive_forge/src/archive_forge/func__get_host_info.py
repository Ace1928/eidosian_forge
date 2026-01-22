from __future__ import absolute_import, division, print_function
import json
import logging
import optparse
import os
import ssl
import sys
import time
from ansible.module_utils.common._collections_compat import MutableMapping
from ansible.module_utils.six import integer_types, text_type, string_types
from ansible.module_utils.six.moves import configparser
from psphere.client import Client
from psphere.errors import ObjectNotFoundError
from psphere.managedobjects import HostSystem, VirtualMachine, ManagedObject, ClusterComputeResource
from suds.sudsobject import Object as SudsObject
def _get_host_info(self, host, prefix='vmware'):
    """
        Return a flattened dict with info about the given host system.
        """
    host_info = {'name': host.name}
    for attr in ('datastore', 'network', 'vm'):
        try:
            value = getattr(host, attr)
            host_info['%ss' % attr] = self._get_obj_info(value, depth=0)
        except AttributeError:
            host_info['%ss' % attr] = []
    for k, v in self._get_obj_info(host.summary, depth=0).items():
        if isinstance(v, MutableMapping):
            for k2, v2 in v.items():
                host_info[k2] = v2
        elif k != 'host':
            host_info[k] = v
    try:
        host_info['ipAddress'] = host.config.network.vnic[0].spec.ip.ipAddress
    except Exception as e:
        print(e, file=sys.stderr)
    host_info = self._flatten_dict(host_info, prefix)
    if '%s_ipAddress' % prefix in host_info:
        host_info['ansible_ssh_host'] = host_info['%s_ipAddress' % prefix]
    return host_info