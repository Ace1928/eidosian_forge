from __future__ import (absolute_import, division, print_function)
import hashlib
import json
import re
import uuid
import os
from collections import namedtuple
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible.module_utils.six import iteritems
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMAuth
from ansible.errors import AnsibleParserError, AnsibleError
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils._text import to_native, to_bytes, to_text
from itertools import chain
class AzureHost(object):
    _powerstate_regex = re.compile('^PowerState/(?P<powerstate>.+)$')

    def __init__(self, vm_model, inventory_client, vmss=None, legacy_name=False):
        self._inventory_client = inventory_client
        self._vm_model = vm_model
        self._vmss = vmss
        self._instanceview = None
        self._powerstate = 'unknown'
        self.nics = []
        if legacy_name:
            self.default_inventory_hostname = vm_model['name']
        else:
            self.default_inventory_hostname = '{0}_{1}'.format(vm_model['name'], hashlib.sha1(to_bytes(vm_model['id'])).hexdigest()[0:4])
        self._hostvars = {}
        inventory_client._enqueue_get(url='{0}/instanceView'.format(vm_model['id']), api_version=self._inventory_client._compute_api_version, handler=self._on_instanceview_response)
        nic_refs = vm_model['properties']['networkProfile']['networkInterfaces']
        for nic in nic_refs:
            is_primary = nic.get('properties', {}).get('primary', len(nic_refs) == 1)
            inventory_client._enqueue_get(url=nic['id'], api_version=self._inventory_client._network_api_version, handler=self._on_nic_response, handler_args=dict(is_primary=is_primary))

    @property
    def hostvars(self):
        if self._hostvars != {}:
            return self._hostvars
        system = 'unknown'
        if 'osProfile' in self._vm_model['properties']:
            if 'linuxConfiguration' in self._vm_model['properties']['osProfile']:
                system = 'linux'
            if 'windowsConfiguration' in self._vm_model['properties']['osProfile']:
                system = 'windows'
        else:
            osType = self._vm_model['properties']['storageProfile']['osDisk']['osType']
            if osType == 'Linux':
                system = 'linux'
            if osType == 'Windows':
                system = 'windows'
        av_zone = None
        if 'zones' in self._vm_model:
            av_zone = self._vm_model['zones']
        new_hostvars = dict(network_interface=[], mac_address=[], network_interface_id=[], security_group_id=[], security_group=[], public_ipv4_addresses=[], public_dns_hostnames=[], private_ipv4_addresses=[], id=self._vm_model['id'], location=self._vm_model['location'], name=self._vm_model['name'], computer_name=self._vm_model['properties'].get('osProfile', {}).get('computerName'), availability_zone=av_zone, powerstate=self._powerstate, provisioning_state=self._vm_model['properties']['provisioningState'].lower(), tags=self._vm_model.get('tags', {}), resource_type=self._vm_model.get('type', 'unknown'), vmid=self._vm_model['properties']['vmId'], os_profile=dict(system=system), vmss=dict(id=self._vmss['id'], name=self._vmss['name']) if self._vmss else {}, virtual_machine_size=self._vm_model['properties']['hardwareProfile']['vmSize'] if self._vm_model['properties'].get('hardwareProfile') else None, plan=self._vm_model['properties']['plan']['name'] if self._vm_model['properties'].get('plan') else None, resource_group=parse_resource_id(self._vm_model['id']).get('resource_group').lower(), default_inventory_hostname=self.default_inventory_hostname, creation_time=self._vm_model['properties']['timeCreated'])
        for nic in sorted(self.nics, key=lambda n: n.is_primary, reverse=True):
            for ipc in sorted(nic._nic_model['properties']['ipConfigurations'], key=lambda i: i['properties'].get('primary', False), reverse=True):
                private_ip = ipc['properties'].get('privateIPAddress')
                if private_ip:
                    new_hostvars['private_ipv4_addresses'].append(private_ip)
                pip_id = ipc['properties'].get('publicIPAddress', {}).get('id')
                if pip_id:
                    new_hostvars['public_ip_id'] = pip_id
                    pip = nic.public_ips[pip_id]
                    new_hostvars['public_ip_name'] = pip._pip_model['name']
                    new_hostvars['public_ipv4_addresses'].append(pip._pip_model['properties'].get('ipAddress', None))
                    pip_fqdn = pip._pip_model['properties'].get('dnsSettings', {}).get('fqdn')
                    if pip_fqdn:
                        new_hostvars['public_dns_hostnames'].append(pip_fqdn)
            new_hostvars['mac_address'].append(nic._nic_model['properties'].get('macAddress'))
            new_hostvars['network_interface'].append(nic._nic_model['name'])
            new_hostvars['network_interface_id'].append(nic._nic_model['id'])
            new_hostvars['security_group_id'].append(nic._nic_model['properties']['networkSecurityGroup']['id']) if nic._nic_model['properties'].get('networkSecurityGroup') else None
            new_hostvars['security_group'].append(parse_resource_id(nic._nic_model['properties']['networkSecurityGroup']['id'])['resource_name']) if nic._nic_model['properties'].get('networkSecurityGroup') else None
        new_hostvars['image'] = {}
        new_hostvars['os_disk'] = {}
        new_hostvars['data_disks'] = []
        storageProfile = self._vm_model['properties'].get('storageProfile')
        if storageProfile:
            imageReference = storageProfile.get('imageReference')
            if imageReference:
                if imageReference.get('publisher'):
                    new_hostvars['image'] = dict(sku=imageReference.get('sku'), publisher=imageReference.get('publisher'), version=imageReference.get('version'), offer=imageReference.get('offer'))
                elif imageReference.get('id'):
                    new_hostvars['image'] = dict(id=imageReference.get('id'))
            osDisk = storageProfile.get('osDisk')
            new_hostvars['os_disk'] = dict(name=osDisk.get('name'), operating_system_type=osDisk.get('osType').lower() if osDisk.get('osType') else None, id=osDisk.get('managedDisk', {}).get('id'))
            new_hostvars['data_disks'] = [dict(name=dataDisk.get('name'), lun=dataDisk.get('lun'), id=dataDisk.get('managedDisk', {}).get('id')) for dataDisk in storageProfile.get('dataDisks', [])]
        self._hostvars = new_hostvars
        return self._hostvars

    def _on_instanceview_response(self, vm_instanceview_model):
        self._instanceview = vm_instanceview_model
        self._powerstate = next((self._powerstate_regex.match(s.get('code', '')).group('powerstate') for s in vm_instanceview_model.get('statuses', []) if self._powerstate_regex.match(s.get('code', ''))), 'unknown')

    def _on_nic_response(self, nic_model, is_primary=False):
        nic = AzureNic(nic_model=nic_model, inventory_client=self._inventory_client, is_primary=is_primary)
        self.nics.append(nic)