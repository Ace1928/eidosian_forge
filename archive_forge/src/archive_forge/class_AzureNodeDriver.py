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
class AzureNodeDriver(NodeDriver):
    connectionCls = AzureServiceManagementConnection
    name = 'Azure Virtual machines'
    website = 'http://azure.microsoft.com/en-us/services/virtual-machines/'
    type = Provider.AZURE
    _instance_types = AZURE_COMPUTE_INSTANCE_TYPES
    _blob_url = '.blob.core.windows.net'
    features = {'create_node': ['password']}
    service_location = collections.namedtuple('service_location', ['is_affinity_group', 'service_location'])
    NODE_STATE_MAP = {'RoleStateUnknown': NodeState.UNKNOWN, 'CreatingVM': NodeState.PENDING, 'StartingVM': NodeState.PENDING, 'Provisioning': NodeState.PENDING, 'CreatingRole': NodeState.PENDING, 'StartingRole': NodeState.PENDING, 'ReadyRole': NodeState.RUNNING, 'BusyRole': NodeState.PENDING, 'StoppingRole': NodeState.PENDING, 'StoppingVM': NodeState.PENDING, 'DeletingVM': NodeState.PENDING, 'StoppedVM': NodeState.STOPPED, 'RestartingRole': NodeState.REBOOTING, 'CyclingRole': NodeState.TERMINATED, 'FailedStartingRole': NodeState.TERMINATED, 'FailedStartingVM': NodeState.TERMINATED, 'UnresponsiveRole': NodeState.TERMINATED, 'StoppedDeallocated': NodeState.TERMINATED}

    def __init__(self, subscription_id=None, key_file=None, **kwargs):
        """
        subscription_id contains the Azure subscription id in the form of GUID
        key_file contains the Azure X509 certificate in .pem form
        """
        self.subscription_id = subscription_id
        self.key_file = key_file
        self.follow_redirects = kwargs.get('follow_redirects', True)
        super().__init__(self.subscription_id, self.key_file, secure=True, **kwargs)

    def list_sizes(self):
        """
        Lists all sizes

        :rtype: ``list`` of :class:`NodeSize`
        """
        sizes = []
        for _, values in self._instance_types.items():
            node_size = self._to_node_size(copy.deepcopy(values))
            sizes.append(node_size)
        return sizes

    def list_images(self, location=None):
        """
        Lists all images

        :rtype: ``list`` of :class:`NodeImage`
        """
        data = self._perform_get(self._get_image_path(), Images)
        custom_image_data = self._perform_get(self._get_vmimage_path(), VMImages)
        images = [self._to_image(i) for i in data]
        images.extend((self._vm_to_image(j) for j in custom_image_data))
        if location is not None:
            images = [image for image in images if location in image.extra['location']]
        return images

    def list_locations(self):
        """
        Lists all locations

        :rtype: ``list`` of :class:`NodeLocation`
        """
        data = self._perform_get('/' + self.subscription_id + '/locations', Locations)
        return [self._to_location(location) for location in data]

    def list_nodes(self, ex_cloud_service_name):
        """
        List all nodes

        ex_cloud_service_name parameter is used to scope the request
        to a specific Cloud Service. This is a required parameter as
        nodes cannot exist outside of a Cloud Service nor be shared
        between a Cloud Service within Azure.

        :param      ex_cloud_service_name: Cloud Service name
        :type       ex_cloud_service_name: ``str``

        :rtype: ``list`` of :class:`Node`
        """
        response = self._perform_get(self._get_hosted_service_path(ex_cloud_service_name) + '?embed-detail=True', None)
        self.raise_for_response(response, 200)
        data = self._parse_response(response, HostedService)
        vips = None
        if len(data.deployments) > 0 and data.deployments[0].virtual_ips is not None:
            vips = [vip.address for vip in data.deployments[0].virtual_ips]
        try:
            return [self._to_node(n, ex_cloud_service_name, vips) for n in data.deployments[0].role_instance_list]
        except IndexError:
            return []

    def reboot_node(self, node, ex_cloud_service_name=None, ex_deployment_slot=None):
        """
        Reboots a node.

        ex_cloud_service_name parameter is used to scope the request
        to a specific Cloud Service. This is a required parameter as
        nodes cannot exist outside of a Cloud Service nor be shared
        between a Cloud Service within Azure.

        :param      ex_cloud_service_name: Cloud Service name
        :type       ex_cloud_service_name: ``str``

        :param      ex_deployment_slot: Options are "production" (default)
                                         or "Staging". (Optional)
        :type       ex_deployment_slot: ``str``

        :rtype: ``bool``
        """
        if ex_cloud_service_name is None:
            if node.extra is not None:
                ex_cloud_service_name = node.extra.get('ex_cloud_service_name')
        if not ex_cloud_service_name:
            raise ValueError('ex_cloud_service_name is required.')
        if not ex_deployment_slot:
            ex_deployment_slot = 'Production'
        _deployment_name = self._get_deployment(service_name=ex_cloud_service_name, deployment_slot=ex_deployment_slot).name
        try:
            response = self._perform_post(self._get_deployment_path_using_name(ex_cloud_service_name, _deployment_name) + '/roleinstances/' + _str(node.id) + '?comp=reboot', '')
            self.raise_for_response(response, 202)
            if self._parse_response_for_async_op(response):
                return True
            else:
                return False
        except Exception:
            return False

    def list_volumes(self, node=None):
        """
        Lists volumes of the disks in the image repository that are
        associated with the specified subscription.

        Pass Node object to scope the list of volumes to a single
        instance.

        :rtype: ``list`` of :class:`StorageVolume`
        """
        data = self._perform_get(self._get_disk_path(), Disks)
        volumes = [self._to_volume(volume=v, node=node) for v in data]
        return volumes

    def create_node(self, name, size, image, ex_cloud_service_name, ex_storage_service_name=None, ex_new_deployment=False, ex_deployment_slot='Production', ex_deployment_name=None, ex_admin_user_id='azureuser', ex_custom_data=None, ex_virtual_network_name=None, ex_network_config=None, auth=None, **kwargs):
        """
        Create Azure Virtual Machine

        Reference: http://bit.ly/1fIsCb7
        [www.windowsazure.com/en-us/documentation/]

        We default to:

        + 3389/TCP - RDP - 1st Microsoft instance.
        + RANDOM/TCP - RDP - All succeeding Microsoft instances.

        + 22/TCP - SSH - 1st Linux instance
        + RANDOM/TCP - SSH - All succeeding Linux instances.

        The above replicates the standard behavior of the Azure UI.
        You can retrieve the assigned ports to each instance by
        using the following private function:

        _get_endpoint_ports(service_name)
        Returns public,private port key pair.

        @inherits: :class:`NodeDriver.create_node`

        :keyword     image: The image to use when creating this node
        :type        image:  `NodeImage`

        :keyword     size: The size of the instance to create
        :type        size: `NodeSize`

        :keyword     ex_cloud_service_name: Required.
                     Name of the Azure Cloud Service.
        :type        ex_cloud_service_name:  ``str``

        :keyword     ex_storage_service_name: Optional:
                     Name of the Azure Storage Service.
        :type        ex_storage_service_name:  ``str``

        :keyword     ex_new_deployment: Optional. Tells azure to create a
                                        new deployment rather than add to an
                                        existing one.
        :type        ex_new_deployment: ``boolean``

        :keyword     ex_deployment_slot: Optional: Valid values: production|
                                         staging.
                                         Defaults to production.
        :type        ex_deployment_slot:  ``str``

        :keyword     ex_deployment_name: Optional. The name of the
                                         deployment.
                                         If this is not passed in we default
                                         to using the Cloud Service name.
        :type        ex_deployment_name: ``str``

        :type        ex_custom_data: ``str``
        :keyword     ex_custom_data: Optional script or other data which is
                                     injected into the VM when it's beginning
                                     provisioned.

        :keyword     ex_admin_user_id: Optional. Defaults to 'azureuser'.
        :type        ex_admin_user_id:  ``str``

        :keyword     ex_virtual_network_name: Optional. If this is not passed
                                              in no virtual network is used.
        :type        ex_virtual_network_name:  ``str``

        :keyword     ex_network_config: Optional. The ConfigurationSet to use
                                        for network configuration
        :type        ex_network_config:  `ConfigurationSet`

        """
        auth = self._get_and_check_auth(auth)
        password = auth.password
        if not isinstance(size, NodeSize):
            raise ValueError('Size must be an instance of NodeSize')
        if not isinstance(image, NodeImage):
            raise ValueError('Image must be an instance of NodeImage, produced by list_images()')
        node_list = self.list_nodes(ex_cloud_service_name=ex_cloud_service_name)
        if ex_network_config is None:
            network_config = ConfigurationSet()
        else:
            network_config = ex_network_config
        network_config.configuration_set_type = 'NetworkConfiguration'
        if ex_custom_data:
            ex_custom_data = self._encode_base64(data=ex_custom_data)
        if WINDOWS_SERVER_REGEX.search(image.id, re.I):
            machine_config = WindowsConfigurationSet(computer_name=name, admin_password=password, admin_user_name=ex_admin_user_id)
            machine_config.domain_join = None
            if not node_list or ex_new_deployment:
                port = '3389'
            else:
                port = random.randint(41952, 65535)
                endpoints = self._get_deployment(service_name=ex_cloud_service_name, deployment_slot=ex_deployment_slot)
                for instances in endpoints.role_instance_list:
                    ports = [ep.public_port for ep in instances.instance_endpoints]
                    while port in ports:
                        port = random.randint(41952, 65535)
            endpoint = ConfigurationSetInputEndpoint(name='Remote Desktop', protocol='tcp', port=port, local_port='3389', load_balanced_endpoint_set_name=None, enable_direct_server_return=False)
        else:
            if not node_list or ex_new_deployment:
                port = '22'
            else:
                port = random.randint(41952, 65535)
                endpoints = self._get_deployment(service_name=ex_cloud_service_name, deployment_slot=ex_deployment_slot)
                for instances in endpoints.role_instance_list:
                    ports = []
                    if instances.instance_endpoints is not None:
                        for ep in instances.instance_endpoints:
                            ports += [ep.public_port]
                    while port in ports:
                        port = random.randint(41952, 65535)
            endpoint = ConfigurationSetInputEndpoint(name='SSH', protocol='tcp', port=port, local_port='22', load_balanced_endpoint_set_name=None, enable_direct_server_return=False)
            machine_config = LinuxConfigurationSet(name, ex_admin_user_id, password, False, ex_custom_data)
        network_config.input_endpoints.items.append(endpoint)
        _storage_location = self._get_cloud_service_location(service_name=ex_cloud_service_name)
        if ex_storage_service_name is None:
            ex_storage_service_name = ex_cloud_service_name
            ex_storage_service_name = re.sub('[\\W_-]+', '', ex_storage_service_name.lower(), flags=re.UNICODE)
            if self._is_storage_service_unique(service_name=ex_storage_service_name):
                self._create_storage_account(service_name=ex_storage_service_name, location=_storage_location.service_location, is_affinity_group=_storage_location.is_affinity_group)
        if not node_list or ex_new_deployment:
            if not ex_deployment_name:
                ex_deployment_name = ex_cloud_service_name
            vm_image_id = None
            disk_config = None
            if image.extra.get('vm_image', False):
                vm_image_id = image.id
            else:
                blob_url = 'http://%s.blob.core.windows.net' % ex_storage_service_name
                disk_name = '{}-{}-{}.vhd'.format(ex_cloud_service_name, name, time.strftime('%Y-%m-%d'))
                media_link = '{}/vhds/{}'.format(blob_url, disk_name)
                disk_config = OSVirtualHardDisk(image.id, media_link)
            response = self._perform_post(self._get_deployment_path_using_name(ex_cloud_service_name), AzureXmlSerializer.virtual_machine_deployment_to_xml(ex_deployment_name, ex_deployment_slot, name, name, machine_config, disk_config, 'PersistentVMRole', network_config, None, None, size.id, ex_virtual_network_name, vm_image_id))
            self.raise_for_response(response, 202)
            self._ex_complete_async_azure_operation(response)
        else:
            _deployment_name = self._get_deployment(service_name=ex_cloud_service_name, deployment_slot=ex_deployment_slot).name
            vm_image_id = None
            disk_config = None
            if image.extra.get('vm_image', False):
                vm_image_id = image.id
            else:
                blob_url = 'http://%s.blob.core.windows.net' % ex_storage_service_name
                disk_name = '{}-{}-{}.vhd'.format(ex_cloud_service_name, name, time.strftime('%Y-%m-%d'))
                media_link = '{}/vhds/{}'.format(blob_url, disk_name)
                disk_config = OSVirtualHardDisk(image.id, media_link)
            path = self._get_role_path(ex_cloud_service_name, _deployment_name)
            body = AzureXmlSerializer.add_role_to_xml(name, machine_config, disk_config, 'PersistentVMRole', network_config, None, None, vm_image_id, size.id)
            response = self._perform_post(path, body)
            self.raise_for_response(response, 202)
            self._ex_complete_async_azure_operation(response)
        return Node(id=name, name=name, state=NodeState.PENDING, public_ips=[], private_ips=[], driver=self.connection.driver, extra={'ex_cloud_service_name': ex_cloud_service_name})

    def destroy_node(self, node, ex_cloud_service_name=None, ex_deployment_slot='Production'):
        """
        Remove Azure Virtual Machine

        This removes the instance, but does not
        remove the disk. You will need to use destroy_volume.
        Azure sometimes has an issue where it will hold onto
        a blob lease for an extended amount of time.

        :keyword     ex_cloud_service_name: Required.
                     Name of the Azure Cloud Service.
        :type        ex_cloud_service_name:  ``str``

        :keyword     ex_deployment_slot: Optional: The name of the deployment
                                         slot. If this is not passed in we
                                         default to production.
        :type        ex_deployment_slot:  ``str``
        """
        if not isinstance(node, Node):
            raise ValueError('A libcloud Node object is required.')
        if ex_cloud_service_name is None and node.extra is not None:
            ex_cloud_service_name = node.extra.get('ex_cloud_service_name')
        if not ex_cloud_service_name:
            raise ValueError('Unable to get ex_cloud_service_name from Node.')
        _deployment = self._get_deployment(service_name=ex_cloud_service_name, deployment_slot=ex_deployment_slot)
        _deployment_name = _deployment.name
        _server_deployment_count = len(_deployment.role_instance_list)
        if _server_deployment_count > 1:
            path = self._get_role_path(ex_cloud_service_name, _deployment_name, node.id)
        else:
            path = self._get_deployment_path_using_name(ex_cloud_service_name, _deployment_name)
        path += '?comp=media'
        self._perform_delete(path)
        return True

    def ex_list_cloud_services(self):
        return self._perform_get(self._get_hosted_service_path(), HostedServices)

    def ex_create_cloud_service(self, name, location, description=None, extended_properties=None):
        """
        Create an azure cloud service.

        :param      name: Name of the service to create
        :type       name: ``str``

        :param      location: Standard azure location string
        :type       location: ``str``

        :param      description: Optional description
        :type       description: ``str``

        :param      extended_properties: Optional extended_properties
        :type       extended_properties: ``dict``

        :rtype: ``bool``
        """
        response = self._perform_cloud_service_create(self._get_hosted_service_path(), AzureXmlSerializer.create_hosted_service_to_xml(name, self._encode_base64(name), description, location, None, extended_properties))
        self.raise_for_response(response, 201)
        return True

    def ex_destroy_cloud_service(self, name):
        """
        Delete an azure cloud service.

        :param      name: Name of the cloud service to destroy.
        :type       name: ``str``

        :rtype: ``bool``
        """
        response = self._perform_cloud_service_delete(self._get_hosted_service_path(name))
        self.raise_for_response(response, 200)
        return True

    def ex_add_instance_endpoints(self, node, endpoints, ex_deployment_slot='Production'):
        all_endpoints = [{'name': endpoint.name, 'protocol': endpoint.protocol, 'port': endpoint.public_port, 'local_port': endpoint.local_port} for endpoint in node.extra['instance_endpoints']]
        all_endpoints.extend(endpoints)
        result = self.ex_set_instance_endpoints(node, all_endpoints, ex_deployment_slot)
        return result

    def ex_set_instance_endpoints(self, node, endpoints, ex_deployment_slot='Production'):
        """
        For example::

            endpoint = ConfigurationSetInputEndpoint(
                name='SSH',
                protocol='tcp',
                port=port,
                local_port='22',
                load_balanced_endpoint_set_name=None,
                enable_direct_server_return=False
            )
            {
                'name': 'SSH',
                'protocol': 'tcp',
                'port': port,
                'local_port': '22'
            }
        """
        ex_cloud_service_name = node.extra['ex_cloud_service_name']
        vm_role_name = node.name
        network_config = ConfigurationSet()
        network_config.configuration_set_type = 'NetworkConfiguration'
        for endpoint in endpoints:
            new_endpoint = ConfigurationSetInputEndpoint(**endpoint)
            network_config.input_endpoints.items.append(new_endpoint)
        _deployment_name = self._get_deployment(service_name=ex_cloud_service_name, deployment_slot=ex_deployment_slot).name
        response = self._perform_put(self._get_role_path(ex_cloud_service_name, _deployment_name, vm_role_name), AzureXmlSerializer.add_role_to_xml(None, None, None, 'PersistentVMRole', network_config, None, None, None, None))
        self.raise_for_response(response, 202)

    def ex_create_storage_service(self, name, location, description=None, affinity_group=None, extended_properties=None):
        """
        Create an azure storage service.

        :param      name: Name of the service to create
        :type       name: ``str``

        :param      location: Standard azure location string
        :type       location: ``str``

        :param      description: (Optional) Description of storage service.
        :type       description: ``str``

        :param      affinity_group: (Optional) Azure affinity group.
        :type       affinity_group: ``str``

        :param      extended_properties: (Optional) Additional configuration
                                         options support by Azure.
        :type       extended_properties: ``dict``

        :rtype: ``bool``
        """
        response = self._perform_storage_service_create(self._get_storage_service_path(), AzureXmlSerializer.create_storage_service_to_xml(service_name=name, label=self._encode_base64(name), description=description, location=location, affinity_group=affinity_group, extended_properties=extended_properties))
        self.raise_for_response(response, 202)
        return True

    def ex_destroy_storage_service(self, name):
        """
        Destroy storage service. Storage service must not have any active
        blobs. Sometimes Azure likes to hold onto volumes after they are
        deleted for an inordinate amount of time, so sleep before calling
        this method after volume deletion.

        :param name: Name of storage service.
        :type  name: ``str``

        :rtype: ``bool``
        """
        response = self._perform_storage_service_delete(self._get_storage_service_path(name))
        self.raise_for_response(response, 200)
        return True
    '\n    Functions not implemented\n    '

    def create_volume_snapshot(self):
        raise NotImplementedError('You cannot create snapshots of Azure VMs at this time.')

    def attach_volume(self):
        raise NotImplementedError('attach_volume is not supported at this time.')

    def create_volume(self):
        raise NotImplementedError('create_volume is not supported at this time.')

    def detach_volume(self):
        raise NotImplementedError('detach_volume is not supported at this time.')

    def destroy_volume(self):
        raise NotImplementedError('destroy_volume is not supported at this time.')
    '\n    Private Functions\n    '

    def _perform_cloud_service_create(self, path, data):
        request = AzureHTTPRequest()
        request.method = 'POST'
        request.host = AZURE_SERVICE_MANAGEMENT_HOST
        request.path = path
        request.body = data
        request.path, request.query = self._update_request_uri_query(request)
        request.headers = self._update_management_header(request)
        response = self._perform_request(request)
        return response

    def _perform_cloud_service_delete(self, path):
        request = AzureHTTPRequest()
        request.method = 'DELETE'
        request.host = AZURE_SERVICE_MANAGEMENT_HOST
        request.path = path
        request.path, request.query = self._update_request_uri_query(request)
        request.headers = self._update_management_header(request)
        response = self._perform_request(request)
        return response

    def _perform_storage_service_create(self, path, data):
        request = AzureHTTPRequest()
        request.method = 'POST'
        request.host = AZURE_SERVICE_MANAGEMENT_HOST
        request.path = path
        request.body = data
        request.path, request.query = self._update_request_uri_query(request)
        request.headers = self._update_management_header(request)
        response = self._perform_request(request)
        return response

    def _perform_storage_service_delete(self, path):
        request = AzureHTTPRequest()
        request.method = 'DELETE'
        request.host = AZURE_SERVICE_MANAGEMENT_HOST
        request.path = path
        request.path, request.query = self._update_request_uri_query(request)
        request.headers = self._update_management_header(request)
        response = self._perform_request(request)
        return response

    def _to_node(self, data, ex_cloud_service_name=None, virtual_ips=None):
        """
        Convert the data from a Azure response object into a Node
        """
        remote_desktop_port = ''
        ssh_port = ''
        public_ips = virtual_ips or []
        if data.instance_endpoints is not None:
            if len(data.instance_endpoints) >= 1:
                public_ips = [data.instance_endpoints[0].vip]
            for port in data.instance_endpoints:
                if port.name == 'Remote Desktop':
                    remote_desktop_port = port.public_port
                if port.name == 'SSH':
                    ssh_port = port.public_port
        return Node(id=data.role_name, name=data.role_name, state=self.NODE_STATE_MAP.get(data.instance_status, NodeState.UNKNOWN), public_ips=public_ips, private_ips=[data.ip_address], driver=self.connection.driver, extra={'instance_endpoints': data.instance_endpoints, 'remote_desktop_port': remote_desktop_port, 'ssh_port': ssh_port, 'power_state': data.power_state, 'instance_size': data.instance_size, 'ex_cloud_service_name': ex_cloud_service_name})

    def _to_location(self, data):
        """
        Convert the data from a Azure response object into a location
        """
        country = data.display_name
        if 'Asia' in data.display_name:
            country = 'Asia'
        if 'Europe' in data.display_name:
            country = 'Europe'
        if 'US' in data.display_name:
            country = 'US'
        if 'Japan' in data.display_name:
            country = 'Japan'
        if 'Brazil' in data.display_name:
            country = 'Brazil'
        vm_role_sizes = data.compute_capabilities.virtual_machines_role_sizes
        return AzureNodeLocation(id=data.name, name=data.display_name, country=country, driver=self.connection.driver, available_services=data.available_services, virtual_machine_role_sizes=vm_role_sizes)

    def _to_node_size(self, data):
        """
        Convert the AZURE_COMPUTE_INSTANCE_TYPES into NodeSize
        """
        return NodeSize(id=data['id'], name=data['name'], ram=data['ram'], disk=data['disk'], bandwidth=data['bandwidth'], price=data['price'], driver=self.connection.driver, extra={'max_data_disks': data['max_data_disks'], 'cores': data['cores']})

    def _to_image(self, data):
        return NodeImage(id=data.name, name=data.label, driver=self.connection.driver, extra={'os': data.os, 'category': data.category, 'description': data.description, 'location': data.location, 'affinity_group': data.affinity_group, 'media_link': data.media_link, 'vm_image': False})

    def _vm_to_image(self, data):
        return NodeImage(id=data.name, name=data.label, driver=self.connection.driver, extra={'os': data.os_disk_configuration.os, 'category': data.category, 'location': data.location, 'media_link': data.os_disk_configuration.media_link, 'affinity_group': data.affinity_group, 'deployment_name': data.deployment_name, 'vm_image': True})

    def _to_volume(self, volume, node):
        extra = {'affinity_group': volume.affinity_group, 'os': volume.os, 'location': volume.location, 'media_link': volume.media_link, 'source_image_name': volume.source_image_name}
        role_name = getattr(volume.attached_to, 'role_name', None)
        hosted_service_name = getattr(volume.attached_to, 'hosted_service_name', None)
        deployment_name = getattr(volume.attached_to, 'deployment_name', None)
        if role_name is not None:
            extra['role_name'] = role_name
        if hosted_service_name is not None:
            extra['hosted_service_name'] = hosted_service_name
        if deployment_name is not None:
            extra['deployment_name'] = deployment_name
        if node:
            if role_name is not None and role_name == node.id:
                return StorageVolume(id=volume.name, name=volume.name, size=int(volume.logical_disk_size_in_gb), driver=self.connection.driver, extra=extra)
        else:
            return StorageVolume(id=volume.name, name=volume.name, size=int(volume.logical_disk_size_in_gb), driver=self.connection.driver, extra=extra)

    def _get_deployment(self, **kwargs):
        _service_name = kwargs['service_name']
        _deployment_slot = kwargs['deployment_slot']
        response = self._perform_get(self._get_deployment_path_using_slot(_service_name, _deployment_slot), None)
        self.raise_for_response(response, 200)
        return self._parse_response(response, Deployment)

    def _get_cloud_service_location(self, service_name=None):
        if not service_name:
            raise ValueError('service_name is required.')
        res = self._perform_get('%s?embed-detail=False' % self._get_hosted_service_path(service_name), HostedService)
        _affinity_group = res.hosted_service_properties.affinity_group
        _cloud_service_location = res.hosted_service_properties.location
        if _affinity_group is not None and _affinity_group != '':
            return self.service_location(True, _affinity_group)
        elif _cloud_service_location is not None:
            return self.service_location(False, _cloud_service_location)
        else:
            return None

    def _is_storage_service_unique(self, service_name=None):
        if not service_name:
            raise ValueError('service_name is required.')
        _check_availability = self._perform_get('%s/operations/isavailable/%s%s' % (self._get_storage_service_path(), _str(service_name), ''), AvailabilityResponse)
        self.raise_for_response(_check_availability, 200)
        return _check_availability.result

    def _create_storage_account(self, **kwargs):
        if kwargs['is_affinity_group'] is True:
            response = self._perform_post(self._get_storage_service_path(), AzureXmlSerializer.create_storage_service_input_to_xml(kwargs['service_name'], kwargs['service_name'], self._encode_base64(kwargs['service_name']), kwargs['location'], None, True, None))
            self.raise_for_response(response, 202)
        else:
            response = self._perform_post(self._get_storage_service_path(), AzureXmlSerializer.create_storage_service_input_to_xml(kwargs['service_name'], kwargs['service_name'], self._encode_base64(kwargs['service_name']), None, kwargs['location'], True, None))
            self.raise_for_response(response, 202)
        self._ex_complete_async_azure_operation(response, 'create_storage_account')

    def _get_operation_status(self, request_id):
        return self._perform_get('/' + self.subscription_id + '/operations/' + _str(request_id), Operation)

    def _perform_get(self, path, response_type):
        request = AzureHTTPRequest()
        request.method = 'GET'
        request.host = AZURE_SERVICE_MANAGEMENT_HOST
        request.path = path
        request.path, request.query = self._update_request_uri_query(request)
        request.headers = self._update_management_header(request)
        response = self._perform_request(request)
        if response_type is not None:
            return self._parse_response(response, response_type)
        return response

    def _perform_post(self, path, body, response_type=None):
        request = AzureHTTPRequest()
        request.method = 'POST'
        request.host = AZURE_SERVICE_MANAGEMENT_HOST
        request.path = path
        request.body = ensure_string(self._get_request_body(body))
        request.path, request.query = self._update_request_uri_query(request)
        request.headers = self._update_management_header(request)
        response = self._perform_request(request)
        return response

    def _perform_put(self, path, body, response_type=None):
        request = AzureHTTPRequest()
        request.method = 'PUT'
        request.host = AZURE_SERVICE_MANAGEMENT_HOST
        request.path = path
        request.body = ensure_string(self._get_request_body(body))
        request.path, request.query = self._update_request_uri_query(request)
        request.headers = self._update_management_header(request)
        response = self._perform_request(request)
        return response

    def _perform_delete(self, path):
        request = AzureHTTPRequest()
        request.method = 'DELETE'
        request.host = AZURE_SERVICE_MANAGEMENT_HOST
        request.path = path
        request.path, request.query = self._update_request_uri_query(request)
        request.headers = self._update_management_header(request)
        response = self._perform_request(request)
        self.raise_for_response(response, 202)

    def _perform_request(self, request):
        try:
            return self.connection.request(action=request.path, data=request.body, headers=request.headers, method=request.method)
        except AzureRedirectException as e:
            parsed_url = urlparse.urlparse(e.location)
            request.host = parsed_url.netloc
            return self._perform_request(request)
        except Exception as e:
            raise e

    def _update_request_uri_query(self, request):
        """
        pulls the query string out of the URI and moves it into
        the query portion of the request object.  If there are already
        query parameters on the request the parameters in the URI will
        appear after the existing parameters
        """
        if '?' in request.path:
            request.path, _, query_string = request.path.partition('?')
            if query_string:
                query_params = query_string.split('&')
                for query in query_params:
                    if '=' in query:
                        name, _, value = query.partition('=')
                        request.query.append((name, value))
        request.path = url_quote(request.path, "/()$=',")
        if request.query:
            request.path += '?'
            for name, value in request.query:
                if value is not None:
                    request.path += '{}={}{}'.format(name, url_quote(value, "/()$=',"), '&')
            request.path = request.path[:-1]
        return (request.path, request.query)

    def _update_management_header(self, request):
        """
        Add additional headers for management.
        """
        if request.method in ['PUT', 'POST', 'MERGE', 'DELETE']:
            request.headers['Content-Length'] = str(len(request.body))
        if request.method not in ['GET', 'HEAD']:
            for key in request.headers:
                if 'content-type' == key.lower():
                    break
            else:
                request.headers['Content-Type'] = 'application/xml'
        return request.headers

    def _parse_response(self, response, return_type):
        """
        Parse the HTTPResponse's body and fill all the data into a class of
        return_type.
        """
        return self._parse_response_body_from_xml_text(response=response, return_type=return_type)

    def _parse_response_body_from_xml_text(self, response, return_type):
        """
        parse the xml and fill all the data into a class of return_type
        """
        respbody = response.body
        doc = minidom.parseString(respbody)
        return_obj = return_type()
        for node in self._get_child_nodes(doc, return_type.__name__):
            self._fill_data_to_return_object(node, return_obj)
        return_obj.status = response.status
        return return_obj

    def _get_child_nodes(self, node, tag_name):
        return [childNode for childNode in node.getElementsByTagName(tag_name) if childNode.parentNode == node]

    def _fill_data_to_return_object(self, node, return_obj):
        members = dict(vars(return_obj))
        for name, value in members.items():
            if isinstance(value, _ListOf):
                setattr(return_obj, name, self._fill_list_of(node, value.list_type, value.xml_element_name))
            elif isinstance(value, ScalarListOf):
                setattr(return_obj, name, self._fill_scalar_list_of(node, value.list_type, self._get_serialization_name(name), value.xml_element_name))
            elif isinstance(value, _DictOf):
                setattr(return_obj, name, self._fill_dict_of(node, self._get_serialization_name(name), value.pair_xml_element_name, value.key_xml_element_name, value.value_xml_element_name))
            elif isinstance(value, WindowsAzureData):
                setattr(return_obj, name, self._fill_instance_child(node, name, value.__class__))
            elif isinstance(value, dict):
                setattr(return_obj, name, self._fill_dict(node, self._get_serialization_name(name)))
            elif isinstance(value, _Base64String):
                value = self._fill_data_minidom(node, name, '')
                if value is not None:
                    value = self._decode_base64_to_text(value)
                setattr(return_obj, name, value)
            else:
                value = self._fill_data_minidom(node, name, value)
                if value is not None:
                    setattr(return_obj, name, value)

    def _fill_list_of(self, xmldoc, element_type, xml_element_name):
        xmlelements = self._get_child_nodes(xmldoc, xml_element_name)
        return [self._parse_response_body_from_xml_node(xmlelement, element_type) for xmlelement in xmlelements]

    def _parse_response_body_from_xml_node(self, node, return_type):
        """
        parse the xml and fill all the data into a class of return_type
        """
        return_obj = return_type()
        self._fill_data_to_return_object(node, return_obj)
        return return_obj

    def _fill_scalar_list_of(self, xmldoc, element_type, parent_xml_element_name, xml_element_name):
        xmlelements = self._get_child_nodes(xmldoc, parent_xml_element_name)
        if xmlelements:
            xmlelements = self._get_child_nodes(xmlelements[0], xml_element_name)
            return [self._get_node_value(xmlelement, element_type) for xmlelement in xmlelements]

    def _get_node_value(self, xmlelement, data_type):
        value = xmlelement.firstChild.nodeValue
        if data_type is datetime:
            return self._to_datetime(value)
        elif data_type is bool:
            return value.lower() != 'false'
        else:
            return data_type(value)

    def _get_serialization_name(self, element_name):
        """
        Converts a Python name into a serializable name.
        """
        known = _KNOWN_SERIALIZATION_XFORMS.get(element_name)
        if known is not None:
            return known
        if element_name.startswith('x_ms_'):
            return element_name.replace('_', '-')
        if element_name.endswith('_id'):
            element_name = element_name.replace('_id', 'ID')
        for name in ['content_', 'last_modified', 'if_', 'cache_control']:
            if element_name.startswith(name):
                element_name = element_name.replace('_', '-_')
        return ''.join((name.capitalize() for name in element_name.split('_')))

    def _fill_dict_of(self, xmldoc, parent_xml_element_name, pair_xml_element_name, key_xml_element_name, value_xml_element_name):
        return_obj = {}
        xmlelements = self._get_child_nodes(xmldoc, parent_xml_element_name)
        if xmlelements:
            xmlelements = self._get_child_nodes(xmlelements[0], pair_xml_element_name)
            for pair in xmlelements:
                keys = self._get_child_nodes(pair, key_xml_element_name)
                values = self._get_child_nodes(pair, value_xml_element_name)
                if keys and values:
                    key = keys[0].firstChild.nodeValue
                    value = values[0].firstChild.nodeValue
                    return_obj[key] = value
        return return_obj

    def _fill_instance_child(self, xmldoc, element_name, return_type):
        """
        Converts a child of the current dom element to the specified type.
        """
        xmlelements = self._get_child_nodes(xmldoc, self._get_serialization_name(element_name))
        if not xmlelements:
            return None
        return_obj = return_type()
        self._fill_data_to_return_object(xmlelements[0], return_obj)
        return return_obj

    def _fill_dict(self, xmldoc, element_name):
        xmlelements = self._get_child_nodes(xmldoc, element_name)
        if xmlelements:
            return_obj = {}
            for child in xmlelements[0].childNodes:
                if child.firstChild:
                    return_obj[child.nodeName] = child.firstChild.nodeValue
            return return_obj

    def _encode_base64(self, data):
        if isinstance(data, _unicode_type):
            data = data.encode('utf-8')
        encoded = base64.b64encode(data)
        return encoded.decode('utf-8')

    def _decode_base64_to_bytes(self, data):
        if isinstance(data, _unicode_type):
            data = data.encode('utf-8')
        return base64.b64decode(data)

    def _decode_base64_to_text(self, data):
        decoded_bytes = self._decode_base64_to_bytes(data)
        return decoded_bytes.decode('utf-8')

    def _fill_data_minidom(self, xmldoc, element_name, data_member):
        xmlelements = self._get_child_nodes(xmldoc, self._get_serialization_name(element_name))
        if not xmlelements or not xmlelements[0].childNodes:
            return None
        value = xmlelements[0].firstChild.nodeValue
        if data_member is None:
            return value
        elif isinstance(data_member, datetime):
            return self._to_datetime(value)
        elif type(data_member) is bool:
            return value.lower() != 'false'
        elif type(data_member) is str:
            return _real_unicode(value)
        else:
            return type(data_member)(value)

    def _to_datetime(self, strtime):
        return datetime.strptime(strtime, '%Y-%m-%dT%H:%M:%S.%f')

    def _get_request_body(self, request_body):
        if request_body is None:
            return b''
        if isinstance(request_body, WindowsAzureData):
            request_body = self._convert_class_to_xml(request_body)
        if isinstance(request_body, bytes):
            return request_body
        if isinstance(request_body, _unicode_type):
            return request_body.encode('utf-8')
        request_body = str(request_body)
        if isinstance(request_body, _unicode_type):
            return request_body.encode('utf-8')
        return request_body

    def _convert_class_to_xml(self, source, xml_prefix=True):
        root = ET.Element()
        doc = self._construct_element_tree(source, root)
        result = ensure_string(ET.tostring(doc, encoding='utf-8', method='xml'))
        return result

    def _construct_element_tree(self, source, etree):
        if source is None:
            return ET.Element()
        if isinstance(source, list):
            for value in source:
                etree.append(self._construct_element_tree(value, etree))
        elif isinstance(source, WindowsAzureData):
            class_name = source.__class__.__name__
            etree.append(ET.Element(class_name))
            for name, value in vars(source).items():
                if value is not None:
                    if isinstance(value, list) or isinstance(value, WindowsAzureData):
                        etree.append(self._construct_element_tree(value, etree))
                    else:
                        ele = ET.Element(self._get_serialization_name(name))
                        ele.text = xml_escape(str(value))
                        etree.append(ele)
            etree.append(ET.Element(class_name))
        return etree

    def _parse_response_for_async_op(self, response):
        if response is None:
            return None
        result = AsynchronousOperationResult()
        if response.headers:
            for name, value in response.headers.items():
                if name.lower() == 'x-ms-request-id':
                    result.request_id = value
        return result

    def _get_deployment_path_using_name(self, service_name, deployment_name=None):
        components = ['services/hostedservices/', _str(service_name), '/deployments']
        resource = ''.join(components)
        return self._get_path(resource, deployment_name)

    def _get_path(self, resource, name):
        path = '/' + self.subscription_id + '/' + resource
        if name is not None:
            path += '/' + _str(name)
        return path

    def _get_image_path(self, image_name=None):
        return self._get_path('services/images', image_name)

    def _get_vmimage_path(self, image_name=None):
        return self._get_path('services/vmimages', image_name)

    def _get_hosted_service_path(self, service_name=None):
        return self._get_path('services/hostedservices', service_name)

    def _get_deployment_path_using_slot(self, service_name, slot=None):
        return self._get_path('services/hostedservices/%s/deploymentslots' % _str(service_name), slot)

    def _get_disk_path(self, disk_name=None):
        return self._get_path('services/disks', disk_name)

    def _get_role_path(self, service_name, deployment_name, role_name=None):
        components = ['services/hostedservices/', _str(service_name), '/deployments/', deployment_name, '/roles']
        resource = ''.join(components)
        return self._get_path(resource, role_name)

    def _get_storage_service_path(self, service_name=None):
        return self._get_path('services/storageservices', service_name)

    def _ex_complete_async_azure_operation(self, response=None, operation_type='create_node'):
        request_id = self._parse_response_for_async_op(response)
        operation_status = self._get_operation_status(request_id.request_id)
        timeout = 60 * 5
        waittime = 0
        interval = 5
        while operation_status.status == 'InProgress' and waittime < timeout:
            operation_status = self._get_operation_status(request_id)
            if operation_status.status == 'Succeeded':
                break
            waittime += interval
            time.sleep(interval)
        if operation_status.status == 'Failed':
            raise LibcloudError('Message: Async request for operation %s has failed' % operation_type, driver=self.connection.driver)

    def raise_for_response(self, response, valid_response):
        if response.status != valid_response:
            values = (response.error, response.body, response.status)
            message = 'Message: %s, Body: %s, Status code: %s' % values
            raise LibcloudError(message, driver=self)