import re
import copy
import time
import base64
import random
import collections
from xml.dom import minidom
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape
from libcloud.utils.py3 import ET, httplib, urlparse
from libcloud.utils.py3 import urlquote as url_quote
from libcloud.utils.py3 import _real_unicode, ensure_string
from libcloud.utils.misc import ReprMixin
from libcloud.common.azure import AzureRedirectException, AzureServiceManagementConnection
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState
from libcloud.compute.providers import Provider
class AzureXmlSerializer:

    @staticmethod
    def create_storage_service_input_to_xml(service_name, description, label, affinity_group, location, geo_replication_enabled, extended_properties):
        return AzureXmlSerializer.doc_from_data('CreateStorageServiceInput', [('ServiceName', service_name), ('Description', description), ('Label', label), ('AffinityGroup', affinity_group), ('Location', location), ('GeoReplicationEnabled', geo_replication_enabled, _lower)], extended_properties)

    @staticmethod
    def update_storage_service_input_to_xml(description, label, geo_replication_enabled, extended_properties):
        return AzureXmlSerializer.doc_from_data('UpdateStorageServiceInput', [('Description', description), ('Label', label, AzureNodeDriver._encode_base64), ('GeoReplicationEnabled', geo_replication_enabled, _lower)], extended_properties)

    @staticmethod
    def regenerate_keys_to_xml(key_type):
        return AzureXmlSerializer.doc_from_data('RegenerateKeys', [('KeyType', key_type)])

    @staticmethod
    def update_hosted_service_to_xml(label, description, extended_properties):
        return AzureXmlSerializer.doc_from_data('UpdateHostedService', [('Label', label, AzureNodeDriver._encode_base64), ('Description', description)], extended_properties)

    @staticmethod
    def create_hosted_service_to_xml(service_name, label, description, location, affinity_group=None, extended_properties=None):
        if affinity_group:
            return AzureXmlSerializer.doc_from_data('CreateHostedService', [('ServiceName', service_name), ('Label', label), ('Description', description), ('AffinityGroup', affinity_group)], extended_properties)
        return AzureXmlSerializer.doc_from_data('CreateHostedService', [('ServiceName', service_name), ('Label', label), ('Description', description), ('Location', location)], extended_properties)

    @staticmethod
    def create_storage_service_to_xml(service_name, label, description, location, affinity_group, extended_properties=None):
        return AzureXmlSerializer.doc_from_data('CreateStorageServiceInput', [('ServiceName', service_name), ('Label', label), ('Description', description), ('Location', location), ('AffinityGroup', affinity_group)], extended_properties)

    @staticmethod
    def create_deployment_to_xml(name, package_url, label, configuration, start_deployment, treat_warnings_as_error, extended_properties):
        return AzureXmlSerializer.doc_from_data('CreateDeployment', [('Name', name), ('PackageUrl', package_url), ('Label', label, AzureNodeDriver._encode_base64), ('Configuration', configuration), ('StartDeployment', start_deployment, _lower), ('TreatWarningsAsError', treat_warnings_as_error, _lower)], extended_properties)

    @staticmethod
    def swap_deployment_to_xml(production, source_deployment):
        return AzureXmlSerializer.doc_from_data('Swap', [('Production', production), ('SourceDeployment', source_deployment)])

    @staticmethod
    def update_deployment_status_to_xml(status):
        return AzureXmlSerializer.doc_from_data('UpdateDeploymentStatus', [('Status', status)])

    @staticmethod
    def change_deployment_to_xml(configuration, treat_warnings_as_error, mode, extended_properties):
        return AzureXmlSerializer.doc_from_data('ChangeConfiguration', [('Configuration', configuration), ('TreatWarningsAsError', treat_warnings_as_error, _lower), ('Mode', mode)], extended_properties)

    @staticmethod
    def upgrade_deployment_to_xml(mode, package_url, configuration, label, role_to_upgrade, force, extended_properties):
        return AzureXmlSerializer.doc_from_data('UpgradeDeployment', [('Mode', mode), ('PackageUrl', package_url), ('Configuration', configuration), ('Label', label, AzureNodeDriver._encode_base64), ('RoleToUpgrade', role_to_upgrade), ('Force', force, _lower)], extended_properties)

    @staticmethod
    def rollback_upgrade_to_xml(mode, force):
        return AzureXmlSerializer.doc_from_data('RollbackUpdateOrUpgrade', [('Mode', mode), ('Force', force, _lower)])

    @staticmethod
    def walk_upgrade_domain_to_xml(upgrade_domain):
        return AzureXmlSerializer.doc_from_data('WalkUpgradeDomain', [('UpgradeDomain', upgrade_domain)])

    @staticmethod
    def certificate_file_to_xml(data, certificate_format, password):
        return AzureXmlSerializer.doc_from_data('CertificateFile', [('Data', data), ('CertificateFormat', certificate_format), ('Password', password)])

    @staticmethod
    def create_affinity_group_to_xml(name, label, description, location):
        return AzureXmlSerializer.doc_from_data('CreateAffinityGroup', [('Name', name), ('Label', label, AzureNodeDriver._encode_base64), ('Description', description), ('Location', location)])

    @staticmethod
    def update_affinity_group_to_xml(label, description):
        return AzureXmlSerializer.doc_from_data('UpdateAffinityGroup', [('Label', label, AzureNodeDriver._encode_base64), ('Description', description)])

    @staticmethod
    def subscription_certificate_to_xml(public_key, thumbprint, data):
        return AzureXmlSerializer.doc_from_data('SubscriptionCertificate', [('SubscriptionCertificatePublicKey', public_key), ('SubscriptionCertificateThumbprint', thumbprint), ('SubscriptionCertificateData', data)])

    @staticmethod
    def os_image_to_xml(label, media_link, name, os):
        return AzureXmlSerializer.doc_from_data('OSImage', [('Label', label), ('MediaLink', media_link), ('Name', name), ('OS', os)])

    @staticmethod
    def data_virtual_hard_disk_to_xml(host_caching, disk_label, disk_name, lun, logical_disk_size_in_gb, media_link, source_media_link):
        return AzureXmlSerializer.doc_from_data('DataVirtualHardDisk', [('HostCaching', host_caching), ('DiskLabel', disk_label), ('DiskName', disk_name), ('Lun', lun), ('LogicalDiskSizeInGB', logical_disk_size_in_gb), ('MediaLink', media_link), ('SourceMediaLink', source_media_link)])

    @staticmethod
    def disk_to_xml(has_operating_system, label, media_link, name, os):
        return AzureXmlSerializer.doc_from_data('Disk', [('HasOperatingSystem', has_operating_system, _lower), ('Label', label), ('MediaLink', media_link), ('Name', name), ('OS', os)])

    @staticmethod
    def restart_role_operation_to_xml():
        xml = ET.Element('OperationType')
        xml.text = 'RestartRoleOperation'
        doc = AzureXmlSerializer.doc_from_xml('RestartRoleOperation', xml)
        result = ensure_string(ET.tostring(doc, encoding='utf-8'))
        return result

    @staticmethod
    def shutdown_role_operation_to_xml():
        xml = ET.Element('OperationType')
        xml.text = 'ShutdownRoleOperation'
        doc = AzureXmlSerializer.doc_from_xml('ShutdownRoleOperation', xml)
        result = ensure_string(ET.tostring(doc, encoding='utf-8'))
        return result

    @staticmethod
    def start_role_operation_to_xml():
        xml = ET.Element('OperationType')
        xml.text = 'StartRoleOperation'
        doc = AzureXmlSerializer.doc_from_xml('StartRoleOperation', xml)
        result = ensure_string(ET.tostring(doc, encoding='utf-8'))
        return result

    @staticmethod
    def windows_configuration_to_xml(configuration, xml):
        AzureXmlSerializer.data_to_xml([('ConfigurationSetType', configuration.configuration_set_type)], xml)
        AzureXmlSerializer.data_to_xml([('ComputerName', configuration.computer_name)], xml)
        AzureXmlSerializer.data_to_xml([('AdminPassword', configuration.admin_password)], xml)
        AzureXmlSerializer.data_to_xml([('ResetPasswordOnFirstLogon', configuration.reset_password_on_first_logon, _lower)], xml)
        AzureXmlSerializer.data_to_xml([('EnableAutomaticUpdates', configuration.enable_automatic_updates, _lower)], xml)
        AzureXmlSerializer.data_to_xml([('TimeZone', configuration.time_zone)], xml)
        if configuration.domain_join is not None:
            domain = ET.xml('DomainJoin')
            creds = ET.xml('Credentials')
            domain.appemnd(creds)
            xml.append(domain)
            AzureXmlSerializer.data_to_xml([('Domain', configuration.domain_join.credentials.domain)], creds)
            AzureXmlSerializer.data_to_xml([('Username', configuration.domain_join.credentials.username)], creds)
            AzureXmlSerializer.data_to_xml([('Password', configuration.domain_join.credentials.password)], creds)
            AzureXmlSerializer.data_to_xml([('JoinDomain', configuration.domain_join.join_domain)], domain)
            AzureXmlSerializer.data_to_xml([('MachineObjectOU', configuration.domain_join.machine_object_ou)], domain)
        if configuration.stored_certificate_settings is not None:
            cert_settings = ET.Element('StoredCertificateSettings')
            xml.append(cert_settings)
            for cert in configuration.stored_certificate_settings:
                cert_setting = ET.Element('CertificateSetting')
                cert_settings.append(cert_setting)
                cert_setting.append(AzureXmlSerializer.data_to_xml([('StoreLocation', cert.store_location)]))
                AzureXmlSerializer.data_to_xml([('StoreName', cert.store_name)], cert_setting)
                AzureXmlSerializer.data_to_xml([('Thumbprint', cert.thumbprint)], cert_setting)
        AzureXmlSerializer.data_to_xml([('AdminUsername', configuration.admin_user_name)], xml)
        return xml

    @staticmethod
    def linux_configuration_to_xml(configuration, xml):
        AzureXmlSerializer.data_to_xml([('ConfigurationSetType', configuration.configuration_set_type)], xml)
        AzureXmlSerializer.data_to_xml([('HostName', configuration.host_name)], xml)
        AzureXmlSerializer.data_to_xml([('UserName', configuration.user_name)], xml)
        AzureXmlSerializer.data_to_xml([('UserPassword', configuration.user_password)], xml)
        AzureXmlSerializer.data_to_xml([('DisableSshPasswordAuthentication', configuration.disable_ssh_password_authentication, _lower)], xml)
        if configuration.ssh is not None:
            ssh = ET.Element('SSH')
            pkeys = ET.Element('PublicKeys')
            kpairs = ET.Element('KeyPairs')
            ssh.append(pkeys)
            ssh.append(kpairs)
            xml.append(ssh)
            for key in configuration.ssh.public_keys:
                pkey = ET.Element('PublicKey')
                pkeys.append(pkey)
                AzureXmlSerializer.data_to_xml([('Fingerprint', key.fingerprint)], pkey)
                AzureXmlSerializer.data_to_xml([('Path', key.path)], pkey)
            for key in configuration.ssh.key_pairs:
                kpair = ET.Element('KeyPair')
                kpairs.append(kpair)
                AzureXmlSerializer.data_to_xml([('Fingerprint', key.fingerprint)], kpair)
                AzureXmlSerializer.data_to_xml([('Path', key.path)], kpair)
        if configuration.custom_data is not None:
            AzureXmlSerializer.data_to_xml([('CustomData', configuration.custom_data)], xml)
        return xml

    @staticmethod
    def network_configuration_to_xml(configuration, xml):
        AzureXmlSerializer.data_to_xml([('ConfigurationSetType', configuration.configuration_set_type)], xml)
        input_endpoints = ET.Element('InputEndpoints')
        xml.append(input_endpoints)
        for endpoint in configuration.input_endpoints:
            input_endpoint = ET.Element('InputEndpoint')
            input_endpoints.append(input_endpoint)
            AzureXmlSerializer.data_to_xml([('LoadBalancedEndpointSetName', endpoint.load_balanced_endpoint_set_name)], input_endpoint)
            AzureXmlSerializer.data_to_xml([('LocalPort', endpoint.local_port)], input_endpoint)
            AzureXmlSerializer.data_to_xml([('Name', endpoint.name)], input_endpoint)
            AzureXmlSerializer.data_to_xml([('Port', endpoint.port)], input_endpoint)
            if endpoint.load_balancer_probe.path or endpoint.load_balancer_probe.port or endpoint.load_balancer_probe.protocol:
                load_balancer_probe = ET.Element('LoadBalancerProbe')
                input_endpoint.append(load_balancer_probe)
                AzureXmlSerializer.data_to_xml([('Path', endpoint.load_balancer_probe.path)], load_balancer_probe)
                AzureXmlSerializer.data_to_xml([('Port', endpoint.load_balancer_probe.port)], load_balancer_probe)
                AzureXmlSerializer.data_to_xml([('Protocol', endpoint.load_balancer_probe.protocol)], load_balancer_probe)
            AzureXmlSerializer.data_to_xml([('Protocol', endpoint.protocol)], input_endpoint)
            AzureXmlSerializer.data_to_xml([('EnableDirectServerReturn', endpoint.enable_direct_server_return, _lower)], input_endpoint)
        subnet_names = ET.Element('SubnetNames')
        xml.append(subnet_names)
        for name in configuration.subnet_names:
            AzureXmlSerializer.data_to_xml([('SubnetName', name)], subnet_names)
        return xml

    @staticmethod
    def role_to_xml(availability_set_name, data_virtual_hard_disks, network_configuration_set, os_virtual_hard_disk, vm_image_name, role_name, role_size, role_type, system_configuration_set, xml):
        AzureXmlSerializer.data_to_xml([('RoleName', role_name)], xml)
        AzureXmlSerializer.data_to_xml([('RoleType', role_type)], xml)
        config_sets = ET.Element('ConfigurationSets')
        xml.append(config_sets)
        if system_configuration_set is not None:
            config_set = ET.Element('ConfigurationSet')
            config_sets.append(config_set)
            if isinstance(system_configuration_set, WindowsConfigurationSet):
                AzureXmlSerializer.windows_configuration_to_xml(system_configuration_set, config_set)
            elif isinstance(system_configuration_set, LinuxConfigurationSet):
                AzureXmlSerializer.linux_configuration_to_xml(system_configuration_set, config_set)
        if network_configuration_set is not None:
            config_set = ET.Element('ConfigurationSet')
            config_sets.append(config_set)
            AzureXmlSerializer.network_configuration_to_xml(network_configuration_set, config_set)
        if availability_set_name is not None:
            AzureXmlSerializer.data_to_xml([('AvailabilitySetName', availability_set_name)], xml)
        if data_virtual_hard_disks is not None:
            vhds = ET.Element('DataVirtualHardDisks')
            xml.append(vhds)
            for hd in data_virtual_hard_disks:
                vhd = ET.Element('DataVirtualHardDisk')
                vhds.append(vhd)
                AzureXmlSerializer.data_to_xml([('HostCaching', hd.host_caching)], vhd)
                AzureXmlSerializer.data_to_xml([('DiskLabel', hd.disk_label)], vhd)
                AzureXmlSerializer.data_to_xml([('DiskName', hd.disk_name)], vhd)
                AzureXmlSerializer.data_to_xml([('Lun', hd.lun)], vhd)
                AzureXmlSerializer.data_to_xml([('LogicalDiskSizeInGB', hd.logical_disk_size_in_gb)], vhd)
                AzureXmlSerializer.data_to_xml([('MediaLink', hd.media_link)], vhd)
        if os_virtual_hard_disk is not None:
            hd = ET.Element('OSVirtualHardDisk')
            xml.append(hd)
            AzureXmlSerializer.data_to_xml([('HostCaching', os_virtual_hard_disk.host_caching)], hd)
            AzureXmlSerializer.data_to_xml([('DiskLabel', os_virtual_hard_disk.disk_label)], hd)
            AzureXmlSerializer.data_to_xml([('DiskName', os_virtual_hard_disk.disk_name)], hd)
            AzureXmlSerializer.data_to_xml([('MediaLink', os_virtual_hard_disk.media_link)], hd)
            AzureXmlSerializer.data_to_xml([('SourceImageName', os_virtual_hard_disk.source_image_name)], hd)
        if vm_image_name is not None:
            AzureXmlSerializer.data_to_xml([('VMImageName', vm_image_name)], xml)
        if role_size is not None:
            AzureXmlSerializer.data_to_xml([('RoleSize', role_size)], xml)
        return xml

    @staticmethod
    def add_role_to_xml(role_name, system_configuration_set, os_virtual_hard_disk, role_type, network_configuration_set, availability_set_name, data_virtual_hard_disks, vm_image_name, role_size):
        doc = AzureXmlSerializer.doc_from_xml('PersistentVMRole')
        xml = AzureXmlSerializer.role_to_xml(availability_set_name, data_virtual_hard_disks, network_configuration_set, os_virtual_hard_disk, vm_image_name, role_name, role_size, role_type, system_configuration_set, doc)
        result = ensure_string(ET.tostring(xml, encoding='utf-8'))
        return result

    @staticmethod
    def update_role_to_xml(role_name, os_virtual_hard_disk, role_type, network_configuration_set, availability_set_name, data_virtual_hard_disks, vm_image_name, role_size):
        doc = AzureXmlSerializer.doc_from_xml('PersistentVMRole')
        AzureXmlSerializer.role_to_xml(availability_set_name, data_virtual_hard_disks, network_configuration_set, os_virtual_hard_disk, vm_image_name, role_name, role_size, role_type, None, doc)
        result = ensure_string(ET.tostring(doc, encoding='utf-8'))
        return result

    @staticmethod
    def capture_role_to_xml(post_capture_action, target_image_name, target_image_label, provisioning_configuration):
        xml = AzureXmlSerializer.data_to_xml([('OperationType', 'CaptureRoleOperation')])
        AzureXmlSerializer.data_to_xml([('PostCaptureAction', post_capture_action)], xml)
        if provisioning_configuration is not None:
            provisioning_config = ET.Element('ProvisioningConfiguration')
            xml.append(provisioning_config)
            if isinstance(provisioning_configuration, WindowsConfigurationSet):
                AzureXmlSerializer.windows_configuration_to_xml(provisioning_configuration, provisioning_config)
            elif isinstance(provisioning_configuration, LinuxConfigurationSet):
                AzureXmlSerializer.linux_configuration_to_xml(provisioning_configuration, provisioning_config)
        AzureXmlSerializer.data_to_xml([('TargetImageLabel', target_image_label)], xml)
        AzureXmlSerializer.data_to_xml([('TargetImageName', target_image_name)], xml)
        doc = AzureXmlSerializer.doc_from_xml('CaptureRoleOperation', xml)
        result = ensure_string(ET.tostring(doc, encoding='utf-8'))
        return result

    @staticmethod
    def virtual_machine_deployment_to_xml(deployment_name, deployment_slot, label, role_name, system_configuration_set, os_virtual_hard_disk, role_type, network_configuration_set, availability_set_name, data_virtual_hard_disks, role_size, virtual_network_name, vm_image_name):
        doc = AzureXmlSerializer.doc_from_xml('Deployment')
        AzureXmlSerializer.data_to_xml([('Name', deployment_name)], doc)
        AzureXmlSerializer.data_to_xml([('DeploymentSlot', deployment_slot)], doc)
        AzureXmlSerializer.data_to_xml([('Label', label)], doc)
        role_list = ET.Element('RoleList')
        role = ET.Element('Role')
        role_list.append(role)
        doc.append(role_list)
        AzureXmlSerializer.role_to_xml(availability_set_name, data_virtual_hard_disks, network_configuration_set, os_virtual_hard_disk, vm_image_name, role_name, role_size, role_type, system_configuration_set, role)
        if virtual_network_name is not None:
            doc.append(AzureXmlSerializer.data_to_xml([('VirtualNetworkName', virtual_network_name)]))
        result = ensure_string(ET.tostring(doc, encoding='utf-8'))
        return result

    @staticmethod
    def data_to_xml(data, xml=None):
        """
        Creates an xml fragment from the specified data.
           data: Array of tuples, where first: xml element name
                                        second: xml element text
                                        third: conversion function
        """
        for element in data:
            name = element[0]
            val = element[1]
            if len(element) > 2:
                converter = element[2]
            else:
                converter = None
            if val is not None:
                if converter is not None:
                    text = _str(converter(_str(val)))
                else:
                    text = _str(val)
                entry = ET.Element(name)
                entry.text = text
                if xml is not None:
                    xml.append(entry)
                else:
                    return entry
        return xml

    @staticmethod
    def doc_from_xml(document_element_name, inner_xml=None):
        """
        Wraps the specified xml in an xml root element with default azure
        namespaces
        """
        '\n        nsmap = {\n            None: "http://www.w3.org/2001/XMLSchema-instance",\n            "i": "http://www.w3.org/2001/XMLSchema-instance"\n        }\n\n        xml.attrib["xmlns:i"] = "http://www.w3.org/2001/XMLSchema-instance"\n        xml.attrib["xmlns"] = "http://schemas.microsoft.com/windowsazure"\n        '
        xml = ET.Element(document_element_name)
        xml.set('xmlns', 'http://schemas.microsoft.com/windowsazure')
        if inner_xml is not None:
            xml.append(inner_xml)
        return xml

    @staticmethod
    def doc_from_data(document_element_name, data, extended_properties=None):
        doc = AzureXmlSerializer.doc_from_xml(document_element_name)
        AzureXmlSerializer.data_to_xml(data, doc)
        if extended_properties is not None:
            doc.append(AzureXmlSerializer.extended_properties_dict_to_xml_fragment(extended_properties))
        result = ensure_string(ET.tostring(doc, encoding='utf-8'))
        return result

    @staticmethod
    def extended_properties_dict_to_xml_fragment(extended_properties):
        if extended_properties is not None and len(extended_properties) > 0:
            xml = ET.Element('ExtendedProperties')
            for key, val in extended_properties.items():
                extended_property = ET.Element('ExtendedProperty')
                name = ET.Element('Name')
                name.text = _str(key)
                value = ET.Element('Value')
                value.text = _str(val)
                extended_property.append(name)
                extended_property.append(value)
                xml.append(extended_property)
            return xml