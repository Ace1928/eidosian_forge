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
def _get_vm_info(self, vm, prefix='vmware'):
    """
        Return a flattened dict with info about the given virtual machine.
        """
    vm_info = {'name': vm.name}
    for attr in ('datastore', 'network'):
        try:
            value = getattr(vm, attr)
            vm_info['%ss' % attr] = self._get_obj_info(value, depth=0)
        except AttributeError:
            vm_info['%ss' % attr] = []
    try:
        vm_info['resourcePool'] = self._get_obj_info(vm.resourcePool, depth=0)
    except AttributeError:
        vm_info['resourcePool'] = ''
    try:
        vm_info['guestState'] = vm.guest.guestState
    except AttributeError:
        vm_info['guestState'] = ''
    for k, v in self._get_obj_info(vm.summary, depth=0).items():
        if isinstance(v, MutableMapping):
            for k2, v2 in v.items():
                if k2 == 'host':
                    k2 = 'hostSystem'
                vm_info[k2] = v2
        elif k != 'vm':
            vm_info[k] = v
    vm_info = self._flatten_dict(vm_info, prefix)
    if '%s_ipAddress' % prefix in vm_info:
        vm_info['ansible_ssh_host'] = vm_info['%s_ipAddress' % prefix]
    return vm_info