from __future__ import absolute_import, division, print_function
import hashlib
import io
import os
import re
import ssl
import sys
import tarfile
import time
import traceback
import xml.etree.ElementTree as ET
from threading import Thread
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible.module_utils.six.moves.urllib.request import Request, urlopen
from ansible.module_utils.urls import generic_urlparse, open_url, urlparse, urlunparse
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
class VMwareDeployOvf(PyVmomi):

    def __init__(self, module):
        super(VMwareDeployOvf, self).__init__(module)
        self.module = module
        self.params = module.params
        self.handle = None
        self.datastore = None
        self.datacenter = None
        self.resource_pool = None
        self.network_mappings = []
        self.ovf_descriptor = None
        self.tar = None
        self.lease = None
        self.import_spec = None
        self.entity = None

    def get_objects(self):
        self.datacenter = self.find_datacenter_by_name(self.params['datacenter'])
        if self.datacenter is None:
            self.module.fail_json(msg=f"Datacenter '{self.params['datacenter']}' could not be located")
        if self.params['cluster']:
            cluster = self.find_cluster_by_name(self.params['cluster'], datacenter_name=self.datacenter)
            if cluster is None:
                self.module.fail_json(msg=f"Unable to find cluster '{self.params['cluster']}'")
            self.resource_pool = self.find_resource_pool_by_cluster(self.params['resource_pool'], cluster=cluster)
        elif self.params['esxi_hostname']:
            host = self.find_hostsystem_by_name(self.params['esxi_hostname'], datacenter=self.datacenter)
            if host is None:
                self.module.fail_json(msg=f"Unable to find host '{self.params['esxi_hostname']}' in datacenter '{{self.params['datacenter']}}'")
            self.resource_pool = self.find_resource_pool_by_name(self.params['resource_pool'], folder=host.parent)
        else:
            self.resource_pool = self.find_resource_pool_by_name(self.params['resource_pool'], folder=self.datacenter.hostFolder)
        if not self.resource_pool:
            self.module.fail_json(msg=f"Resource pool '{self.params['resource_pool']}' could not be located")
        self.datastore = None
        datastore_cluster_obj = self.find_datastore_cluster_by_name(self.params['datastore'], datacenter=self.datacenter)
        if datastore_cluster_obj:
            datastore = None
            datastore_freespace = 0
            for ds in datastore_cluster_obj.childEntity:
                if isinstance(ds, vim.Datastore) and ds.summary.freeSpace > datastore_freespace:
                    if ds.summary.maintenanceMode != 'normal' or not ds.summary.accessible:
                        continue
                    datastore = ds
                    datastore_freespace = ds.summary.freeSpace
            if datastore:
                self.datastore = datastore
        else:
            self.datastore = self.find_datastore_by_name(self.params['datastore'], datacenter_name=self.datacenter)
        if self.datastore is None:
            self.module.fail_json(msg=f"Datastore '{self.params['datastore']}' could not be located on specified ESXi host or datacenter")
        for key, value in self.params['networks'].items():
            networks = find_all_networks_by_name(self.content, value, datacenter_name=self.datacenter)
            if not networks:
                self.module.fail_json(msg=f"Network '{value}' could not be located")
            for network in networks:
                if self.params['cluster']:
                    if network in cluster.network:
                        network_mapping = vim.OvfManager.NetworkMapping()
                        network_mapping.name = key
                        network_mapping.network = network
                        self.network_mappings.append(network_mapping)
                else:
                    network_mapping = vim.OvfManager.NetworkMapping()
                    network_mapping.name = key
                    network_mapping.network = network
                    self.network_mappings.append(network_mapping)
        return (self.datastore, self.datacenter, self.resource_pool, self.network_mappings)

    def get_ovf_descriptor(self):
        if self.params['url'] is None:
            try:
                path_exists(self.params['ovf'])
            except ValueError as e:
                self.module.fail_json(msg='%s' % e)
            if tarfile.is_tarfile(self.params['ovf']):
                self.tar = tarfile.open(self.params['ovf'])
                ovf = None
                for candidate in self.tar.getmembers():
                    dummy, ext = os.path.splitext(candidate.name)
                    if ext.lower() == '.ovf':
                        ovf = candidate
                        break
                if not ovf:
                    self.module.fail_json(msg='Could not locate OVF file in %(ovf)s' % self.params)
                self.ovf_descriptor = to_native(self.tar.extractfile(ovf).read())
            else:
                with open(self.params['ovf'], encoding='utf-8') as f:
                    self.ovf_descriptor = f.read()
            return self.ovf_descriptor
        else:
            self.handle = WebHandle(self.params['url'])
            self.tar = tarfile.open(fileobj=self.handle)
            ovffilename = list(filter(lambda x: x.endswith('.ovf'), self.tar.getnames()))[0]
            ovffile = self.tar.extractfile(ovffilename)
            self.ovf_descriptor = ovffile.read().decode()
            if self.ovf_descriptor:
                return self.ovf_descriptor
            else:
                self.module.fail_json(msg='Could not locate OVF file in %(url)s' % self.params)

    def get_lease(self):
        datastore, datacenter, resource_pool, network_mappings = self.get_objects()
        params = {'diskProvisioning': self.params['disk_provisioning']}
        if self.params['name']:
            params['entityName'] = self.params['name']
        if network_mappings:
            params['networkMapping'] = network_mappings
        if self.params['deployment_option']:
            params['deploymentOption'] = self.params['deployment_option']
        if self.params['properties']:
            params['propertyMapping'] = []
            for key, value in self.params['properties'].items():
                property_mapping = vim.KeyValue()
                property_mapping.key = key
                property_mapping.value = str(value) if isinstance(value, bool) else value
                params['propertyMapping'].append(property_mapping)
        if self.params['folder']:
            folder = self.content.searchIndex.FindByInventoryPath(self.params['folder'])
            if not folder:
                self.module.fail_json(msg=f'Unable to find the specified folder {self.params['folder']}')
        else:
            folder = datacenter.vmFolder
        spec_params = vim.OvfManager.CreateImportSpecParams(**params)
        ovf_descriptor = self.get_ovf_descriptor()
        self.import_spec = self.content.ovfManager.CreateImportSpec(ovf_descriptor, resource_pool, datastore, spec_params)
        if self.params['enable_hidden_properties']:
            for prop in self.import_spec.importSpec.configSpec.vAppConfig.property:
                prop.info.userConfigurable = True
        errors = [to_native(e.msg) for e in getattr(self.import_spec, 'error', [])]
        if self.params['fail_on_spec_warnings']:
            errors.extend((to_native(w.msg) for w in getattr(self.import_spec, 'warning', [])))
        if errors:
            self.module.fail_json(msg=f'Failure validating OVF import spec: {'. '.join(errors)}')
        for warning in getattr(self.import_spec, 'warning', []):
            self.module.warn(f'Problem validating OVF import spec: {to_native(warning.msg)}')
        name = self.params.get('name')
        if not self.params['allow_duplicates']:
            name = self.import_spec.importSpec.configSpec.name
            match = find_vm_by_name(self.content, name, folder=folder)
            if match:
                self.module.exit_json(instance=gather_vm_facts(self.content, match), changed=False)
        if self.module.check_mode:
            self.module.exit_json(changed=True, instance={'hw_name': name})
        try:
            self.lease = resource_pool.ImportVApp(self.import_spec.importSpec, folder)
        except vmodl.fault.SystemError as err:
            self.module.fail_json(msg=f'Failed to start import: {to_native(err.msg)}')
        while self.lease.state != vim.HttpNfcLease.State.ready:
            time.sleep(0.1)
        self.entity = self.lease.info.entity
        return (self.lease, self.import_spec)

    def _normalize_url(self, url):
        """
        The hostname in URLs from vmware may be ``*`` update it accordingly
        """
        url_parts = generic_urlparse(urlparse(url))
        if url_parts.hostname == '*':
            if url_parts.port:
                url_parts.netloc = '%s:%d' % (self.params['hostname'], url_parts.port)
            else:
                url_parts.netloc = self.params['hostname']
        return urlunparse(url_parts.as_list())

    def vm_existence_check(self):
        vm_obj = self.get_vm()
        if vm_obj:
            self.entity = vm_obj
            facts = self.deploy()
            self.module.exit_json(**facts)

    def upload(self):
        if self.params['ovf'] is None:
            lease, import_spec = self.get_lease()
            ssl_thumbprint = self.handle.thumbprint if self.handle.thumbprint else None
            source_files = []
            for file_item in import_spec.fileItem:
                source_file = vim.HttpNfcLease.SourceFile(url=self.handle.url, targetDeviceId=file_item.deviceId, create=file_item.create, size=file_item.size, sslThumbprint=ssl_thumbprint, memberName=file_item.path)
                source_files.append(source_file)
            wait_for_task(lease.HttpNfcLeasePullFromUrls_Task(source_files))
        else:
            ovf_dir = os.path.dirname(self.params['ovf'])
            lease, import_spec = self.get_lease()
            uploaders = []
            for file_item in import_spec.fileItem:
                device_upload_url = None
                for device_url in lease.info.deviceUrl:
                    if file_item.deviceId == device_url.importKey:
                        device_upload_url = self._normalize_url(device_url.url)
                        break
                if not device_upload_url:
                    lease.HttpNfcLeaseAbort(vmodl.fault.SystemError(reason=f"Failed to find deviceUrl for file '{file_item.path}'"))
                    self.module.fail_json(msg=f"Failed to find deviceUrl for file '{file_item.path}'")
                vmdk_tarinfo = None
                if self.tar:
                    vmdk = self.tar
                    try:
                        vmdk_tarinfo = self.tar.getmember(file_item.path)
                    except KeyError:
                        lease.HttpNfcLeaseAbort(vmodl.fault.SystemError(reason=f"Failed to find VMDK file '{file_item.path}' in OVA"))
                        self.module.fail_json(msg=f"Failed to find VMDK file '{file_item.path}' in OVA")
                else:
                    vmdk = os.path.join(ovf_dir, file_item.path)
                    try:
                        path_exists(vmdk)
                    except ValueError:
                        lease.HttpNfcLeaseAbort(vmodl.fault.SystemError(reason=f"Failed to find VMDK file at '{vmdk}'"))
                        self.module.fail_json(msg=f"Failed to find VMDK file at '{vmdk}'")
                uploaders.append(VMDKUploader(vmdk, device_upload_url, self.params['validate_certs'], tarinfo=vmdk_tarinfo, create=file_item.create))
            total_size = sum((u.size for u in uploaders))
            total_bytes_read = [0] * len(uploaders)
            for i, uploader in enumerate(uploaders):
                uploader.start()
                while uploader.is_alive():
                    time.sleep(0.1)
                    total_bytes_read[i] = uploader.bytes_read
                    lease.HttpNfcLeaseProgress(int(100.0 * sum(total_bytes_read) / total_size))
                if uploader.e:
                    lease.HttpNfcLeaseAbort(vmodl.fault.SystemError(reason='%s' % to_native(uploader.e[1])))
                    self.module.fail_json(msg='%s' % to_native(uploader.e[1]), exception=''.join(traceback.format_tb(uploader.e[2])))

    def complete(self):
        self.lease.HttpNfcLeaseComplete()

    def inject_ovf_env(self):
        attrib = {'xmlns': 'http://schemas.dmtf.org/ovf/environment/1', 'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance', 'xmlns:oe': 'http://schemas.dmtf.org/ovf/environment/1', 'xmlns:ve': 'http://www.vmware.com/schema/ovfenv', 'oe:id': '', 've:esxId': self.entity._moId}
        env = ET.Element('Environment', **attrib)
        platform = ET.SubElement(env, 'PlatformSection')
        ET.SubElement(platform, 'Kind').text = self.content.about.name
        ET.SubElement(platform, 'Version').text = self.content.about.version
        ET.SubElement(platform, 'Vendor').text = self.content.about.vendor
        ET.SubElement(platform, 'Locale').text = 'US'
        prop_section = ET.SubElement(env, 'PropertySection')
        for key, value in self.params['properties'].items():
            params = {'oe:key': key, 'oe:value': str(value) if isinstance(value, bool) else value}
            ET.SubElement(prop_section, 'Property', **params)
        opt = vim.option.OptionValue()
        opt.key = 'guestinfo.ovfEnv'
        opt.value = '<?xml version="1.0" encoding="UTF-8"?>' + to_native(ET.tostring(env))
        config_spec = vim.vm.ConfigSpec()
        config_spec.extraConfig = [opt]
        task = self.entity.ReconfigVM_Task(config_spec)
        wait_for_task(task)

    def deploy(self):
        facts = {}
        if self.params['power_on']:
            facts = set_vm_power_state(self.content, self.entity, 'poweredon', force=False)
            if self.params['wait_for_ip_address']:
                _facts = wait_for_vm_ip(self.content, self.entity)
                if not _facts:
                    self.module.fail_json(msg='Waiting for IP address timed out')
        if not facts:
            facts.update(dict(instance=gather_vm_facts(self.content, self.entity)))
        return facts