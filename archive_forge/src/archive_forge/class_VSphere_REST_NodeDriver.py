import ssl
import json
import time
import atexit
import base64
import asyncio
import hashlib
import logging
import warnings
import functools
import itertools
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
class VSphere_REST_NodeDriver(NodeDriver):
    name = 'VMware vSphere'
    website = 'http://www.vmware.com/products/vsphere/'
    type = Provider.VSPHERE
    connectionCls = VSphereConnection
    session_token = None
    NODE_STATE_MAP = {'powered_on': NodeState.RUNNING, 'powered_off': NodeState.STOPPED, 'suspended': NodeState.SUSPENDED}
    VALID_RESPONSE_CODES = [httplib.OK, httplib.ACCEPTED, httplib.CREATED, httplib.NO_CONTENT]

    def __init__(self, key, secret=None, secure=True, host=None, port=443, ca_cert=None):
        if not key or not secret:
            raise InvalidCredsError('Please provide both username (key) and password (secret).')
        super().__init__(key=key, secure=secure, host=host, port=port)
        prefixes = ['http://', 'https://']
        for prefix in prefixes:
            if host.startswith(prefix):
                host = host.lstrip(prefix)
        if ca_cert:
            self.connection.connection.ca_cert = ca_cert
        else:
            self.connection.connection.ca_cert = False
        self.connection.secret = secret
        self.host = host
        self.username = key
        self._get_session_token()
        self.driver_soap = None

    def _get_soap_driver(self):
        if pyvmomi is None:
            raise ImportError('Missing "pyvmomi" dependency. You can install it using pip - pip install pyvmomi')
        self.driver_soap = VSphereNodeDriver(self.host, self.username, self.connection.secret, ca_cert=self.connection.connection.ca_cert)

    def _get_session_token(self):
        uri = '/rest/com/vmware/cis/session'
        try:
            result = self.connection.request(uri, method='POST')
        except Exception:
            raise
        self.session_token = result.object['value']
        self.connection.session_token = self.session_token

    def list_sizes(self):
        return []

    def list_nodes(self, ex_filter_power_states=None, ex_filter_folders=None, ex_filter_names=None, ex_filter_hosts=None, ex_filter_clusters=None, ex_filter_vms=None, ex_filter_datacenters=None, ex_filter_resource_pools=None, max_properties=20):
        """
        The ex parameters are search options and must be an array of strings,
        even ex_filter_power_states which can have at most two items but makes
        sense to keep only one ("POWERED_ON" or "POWERED_OFF")
        Keep in mind that this method will return up to 1000 nodes so if your
        network has more, do use the provided filters and call it multiple
        times.
        """
        req = '/rest/vcenter/vm'
        kwargs = {'filter.power_states': ex_filter_power_states, 'filter.folders': ex_filter_folders, 'filter.names': ex_filter_names, 'filter.hosts': ex_filter_hosts, 'filter.clusters': ex_filter_clusters, 'filter.vms': ex_filter_vms, 'filter.datacenters': ex_filter_datacenters, 'filter.resource_pools': ex_filter_resource_pools}
        params = {}
        for param, value in kwargs.items():
            if value:
                params[param] = value
        result = self._request(req, params=params).object['value']
        vm_ids = [[item['vm']] for item in result]
        vms = []
        interfaces = self._list_interfaces()
        for vm_id in vm_ids:
            vms.append(self._to_node(vm_id, interfaces))
        return vms

    def async_list_nodes(self):
        """
        In this case filtering is not possible.
        Use this method when the cloud has
        a lot of vms and you want to return them all.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self._get_all_vms())
        vm_ids = [(item['vm'], item['host']) for item in result]
        interfaces = self._list_interfaces()
        return loop.run_until_complete(self._list_nodes_async(vm_ids, interfaces))

    async def _list_nodes_async(self, vm_ids, interfaces):
        loop = asyncio.get_event_loop()
        vms = [loop.run_in_executor(None, self._to_node, vm_ids[i], interfaces) for i in range(len(vm_ids))]
        return await asyncio.gather(*vms)

    async def _get_all_vms(self):
        """
        6.7 doesn't offer any pagination, if we get 1000 vms we will try
        this roundabout  way: First get all the datacenters, for each
        datacenter get the hosts and for each host the vms it has.
        This assumes that datacenters, hosts per datacenter and vms per
        host don't exceed 1000.
        """
        datacenters = self.ex_list_datacenters()
        loop = asyncio.get_event_loop()
        hosts_futures = [loop.run_in_executor(None, functools.partial(self.ex_list_hosts, ex_filter_datacenters=datacenter['id'])) for datacenter in datacenters]
        hosts = await asyncio.gather(*hosts_futures)
        vm_resp_futures = [loop.run_in_executor(None, functools.partial(self._get_vms_with_host, host)) for host in itertools.chain(*hosts)]
        vm_resp = await asyncio.gather(*vm_resp_futures)
        return [item for vm_list in vm_resp for item in vm_list]

    def _get_vms_with_host(self, host):
        req = '/rest/vcenter/vm'
        host_id = host['host']
        response = self._request(req, params={'filter.hosts': host_id})
        vms = response.object['value']
        for vm in vms:
            vm['host'] = host
        return vms

    def list_locations(self, ex_show_hosts_in_drs=True):
        """
        Location in the general sense means any resource that allows for node
        creation. In vSphere's case that usually is a host but if a cluster
        has rds enabled, a cluster can be assigned to create the VM, thus the
        clusters with rds enabled will be added to locations.

        :param ex_show_hosts_in_drs: A DRS cluster schedules automatically
                                     on which host should be placed thus it
                                     may not be desired to show the hosts
                                     in a DRS enabled cluster. Set False to
                                     not show these hosts.
        :type ex_show_hosts_in_drs:  `boolean`
        """
        clusters = self.ex_list_clusters()
        hosts_all = self.ex_list_hosts()
        hosts = []
        if ex_show_hosts_in_drs:
            hosts = hosts_all
        else:
            cluster_filter = [cluster['cluster'] for cluster in clusters]
            filter_hosts = self.ex_list_hosts(ex_filter_clusters=cluster_filter)
            hosts = [host for host in hosts_all if host not in filter_hosts]
        driver = self.connection.driver
        locations = []
        for cluster in clusters:
            if cluster['drs_enabled']:
                extra = {'type': 'cluster', 'drs': True, 'ha': cluster['ha_enabled']}
                locations.append(NodeLocation(id=cluster['cluster'], name=cluster['name'], country='', driver=driver, extra=extra))
        for host in hosts:
            extra = {'type': 'host', 'status': host['connection_state'], 'state': host['power_state']}
            locations.append(NodeLocation(id=host['host'], name=host['name'], country='', driver=driver, extra=extra))
        return locations

    def stop_node(self, node):
        if node.state == NodeState.STOPPED:
            return True
        method = 'POST'
        req = '/rest/vcenter/vm/{}/power/stop'.format(node.id)
        result = self._request(req, method=method)
        return result.status in self.VALID_RESPONSE_CODES

    def start_node(self, node):
        if isinstance(node, str):
            node_id = node
        else:
            if node.state is NodeState.RUNNING:
                return True
            node_id = node.id
        method = 'POST'
        req = '/rest/vcenter/vm/{}/power/start'.format(node_id)
        result = self._request(req, method=method)
        return result.status in self.VALID_RESPONSE_CODES

    def reboot_node(self, node):
        if node.state is not NodeState.RUNNING:
            return False
        method = 'POST'
        req = '/rest/vcenter/vm/{}/power/reset'.format(node.id)
        result = self._request(req, method=method)
        return result.status in self.VALID_RESPONSE_CODES

    def destroy_node(self, node):
        if node.state is not NodeState.STOPPED:
            self.stop_node(node)
        req = '/rest/vcenter/vm/{}'.format(node.id)
        resp = self._request(req, method='DELETE')
        return resp.status in self.VALID_RESPONSE_CODES

    def ex_suspend_node(self, node):
        if node.state is not NodeState.RUNNING:
            return False
        method = 'POST'
        req = '/rest/vcenter/vm/{}/power/suspend'.format(node.id)
        result = self._request(req, method=method)
        return result.status in self.VALID_RESPONSE_CODES

    def _list_interfaces(self):
        request = '/rest/appliance/networking/interfaces'
        response = self._request(request).object['value']
        interfaces = [{'name': interface['name'], 'mac': interface['mac'], 'status': interface['status'], 'ip': interface['ipv4']['address']} for interface in response]
        return interfaces

    def _to_node(self, vm_id_host, interfaces):
        """
        id, name, state, public_ips, private_ips,
                driver, size=None, image=None, extra=None, created_at=None)
        """
        vm_id = vm_id_host[0]
        req = '/rest/vcenter/vm/' + vm_id
        vm = self._request(req).object['value']
        name = vm['name']
        state = self.NODE_STATE_MAP[vm['power_state'].lower()]
        private_ips = []
        nic_macs = set()
        for nic in vm['nics']:
            nic_macs.add(nic['value']['mac_address'])
        for interface in interfaces:
            if interface['mac'] in nic_macs:
                private_ips.append(interface['ip'])
                nic_macs.remove(interface['mac'])
                if len(nic_macs) == 0:
                    break
        public_ips = []
        driver = self.connection.driver
        total_size = 0
        gb_converter = 1024 ** 3
        for disk in vm['disks']:
            total_size += int(int(disk['value']['capacity'] / gb_converter))
        ram = int(vm['memory']['size_MiB'])
        cpus = int(vm['cpu']['count'])
        id_to_hash = str(ram) + str(cpus) + str(total_size)
        size_id = hashlib.md5(id_to_hash.encode('utf-8')).hexdigest()
        size_name = name + '-size'
        size_extra = {'cpus': cpus}
        size = NodeSize(id=size_id, name=size_name, ram=ram, disk=total_size, bandwidth=0, price=0, driver=driver, extra=size_extra)
        image_name = vm['guest_OS']
        image_id = image_name + '_id'
        image_extra = {'type': 'guest_OS'}
        image = NodeImage(id=image_id, name=image_name, driver=driver, extra=image_extra)
        extra = {'snapshots': []}
        if len(vm_id_host) > 1:
            extra['host'] = vm_id_host[1].get('name', '')
        return Node(id=vm_id, name=name, state=state, public_ips=public_ips, private_ips=private_ips, driver=driver, size=size, image=image, extra=extra)

    def ex_list_hosts(self, ex_filter_folders=None, ex_filter_standalone=None, ex_filter_hosts=None, ex_filter_clusters=None, ex_filter_names=None, ex_filter_datacenters=None, ex_filter_connection_states=None):
        kwargs = {'filter.folders': ex_filter_folders, 'filter.names': ex_filter_names, 'filter.hosts': ex_filter_hosts, 'filter.clusters': ex_filter_clusters, 'filter.standalone': ex_filter_standalone, 'filter.datacenters': ex_filter_datacenters, 'filter.connection_states': ex_filter_connection_states}
        params = {}
        for param, value in kwargs.items():
            if value:
                params[param] = value
        req = '/rest/vcenter/host'
        result = self._request(req, params=params).object['value']
        return result

    def ex_list_clusters(self, ex_filter_folders=None, ex_filter_names=None, ex_filter_datacenters=None, ex_filter_clusters=None):
        kwargs = {'filter.folders': ex_filter_folders, 'filter.names': ex_filter_names, 'filter.datacenters': ex_filter_datacenters, 'filter.clusters': ex_filter_clusters}
        params = {}
        for param, value in kwargs.items():
            if value:
                params[param] = value
        req = '/rest/vcenter/cluster'
        result = self._request(req, params=params).object['value']
        return result

    def ex_list_datacenters(self, ex_filter_folders=None, ex_filter_names=None, ex_filter_datacenters=None):
        req = '/rest/vcenter/datacenter'
        kwargs = {'filter.folders': ex_filter_folders, 'filter.names': ex_filter_names, 'filter.datacenters': ex_filter_datacenters}
        params = {}
        for param, value in kwargs.items():
            if value:
                params[param] = value
        result = self._request(req, params=params)
        to_return = [{'name': item['name'], 'id': item['datacenter']} for item in result.object['value']]
        return to_return

    def ex_list_content_libraries(self):
        req = '/rest/com/vmware/content/library'
        try:
            result = self._request(req).object
            return result['value']
        except BaseHTTPError:
            return []

    def ex_list_content_library_items(self, library_id):
        req = '/rest/com/vmware/content/library/item'
        params = {'library_id': library_id}
        try:
            result = self._request(req, params=params).object
            return result['value']
        except BaseHTTPError:
            logger.error('Library was cannot be accessed,  most probably the VCenter service is stopped')
            return []

    def ex_list_folders(self):
        req = '/rest/vcenter/folder'
        response = self._request(req).object
        folders = response['value']
        for folder in folders:
            folder['id'] = folder['folder']
        return folders

    def ex_list_datastores(self, ex_filter_folders=None, ex_filter_names=None, ex_filter_datacenters=None, ex_filter_types=None, ex_filter_datastores=None):
        req = '/rest/vcenter/datastore'
        kwargs = {'filter.folders': ex_filter_folders, 'filter.names': ex_filter_names, 'filter.datacenters': ex_filter_datacenters, 'filter.types': ex_filter_types, 'filter.datastores': ex_filter_datastores}
        params = {}
        for param, value in kwargs.items():
            if value:
                params[param] = value
        result = self._request(req, params=params).object['value']
        for datastore in result:
            datastore['id'] = datastore['datastore']
        return result

    def ex_update_memory(self, node, ram):
        """
        :param ram: The amount of ram in MB.
        :type ram: `str` or `int`
        """
        if isinstance(node, str):
            node_id = node
        else:
            node_id = node.id
        request = '/rest/vcenter/vm/{}/hardware/memory'.format(node_id)
        ram = int(ram)
        body = {'spec': {'size_MiB': ram}}
        response = self._request(request, method='PATCH', data=json.dumps(body))
        return response.status in self.VALID_RESPONSE_CODES

    def ex_update_cpu(self, node, cores):
        """
        Assuming 1 Core per socket
        :param cores: Integer or string indicating number of cores
        :type cores: `int` or `str`
        """
        if isinstance(node, str):
            node_id = node
        else:
            node_id = node.id
        request = '/rest/vcenter/vm/{}/hardware/cpu'.format(node_id)
        cores = int(cores)
        body = {'spec': {'count': cores}}
        response = self._request(request, method='PATCH', data=json.dumps(body))
        return response.status in self.VALID_RESPONSE_CODES

    def ex_update_capacity(self, node, capacity):
        pass

    def ex_add_nic(self, node, network):
        """
        Creates a network adapter that will connect to the specified network
        for the given node. Returns a boolean indicating success or not.
        """
        if isinstance(node, str):
            node_id = node
        else:
            node_id = node.id
        spec = {}
        spec['mac_type'] = 'GENERATED'
        spec['backing'] = {}
        spec['backing']['type'] = 'STANDARD_PORTGROUP'
        spec['backing']['network'] = network
        spec['start_connected'] = True
        data = json.dumps({'spec': spec})
        req = '/rest/vcenter/vm/{}/hardware/ethernet'.format(node_id)
        method = 'POST'
        resp = self._request(req, method=method, data=data)
        return resp.status

    def _get_library_item(self, item_id):
        req = '/rest/com/vmware/content/library/item/id:{}'.format(item_id)
        result = self._request(req).object
        return result['value']

    def _get_resource_pool(self, host_id=None, cluster_id=None, name=None):
        if host_id:
            pms = {'filter.hosts': host_id}
        if cluster_id:
            pms = {'filter.clusters': cluster_id}
        if name:
            pms = {'filter.names': name}
        rp_request = '/rest/vcenter/resource-pool'
        resource_pool = self._request(rp_request, params=pms).object
        return resource_pool['value'][0]['resource_pool']

    def _request(self, req, method='GET', params=None, data=None):
        try:
            result = self.connection.request(req, method=method, params=params, data=data)
        except BaseHTTPError as exc:
            if exc.code == 401:
                self.connection.session_token = None
                self._get_session_token()
                result = self.connection.request(req, method=method, params=params, data=data)
            else:
                raise
        except Exception:
            raise
        return result

    def list_images(self, **kwargs):
        libraries = self.ex_list_content_libraries()
        item_ids = []
        if libraries:
            for library in libraries:
                item_ids.extend(self.ex_list_content_library_items(library))
        items = []
        if item_ids:
            for item_id in item_ids:
                items.append(self._get_library_item(item_id))
        images = []
        names = set()
        if items:
            driver = self.connection.driver
            for item in items:
                names.add(item['name'])
                extra = {'type': item['type']}
                if item['type'] == 'vm-template':
                    capacity = item['size'] // 1024 ** 3
                    extra['disk_size'] = capacity
                images.append(NodeImage(id=item['id'], name=item['name'], driver=driver, extra=extra))
        if self.driver_soap is None:
            self._get_soap_driver()
        templates_in_hosts = self.driver_soap.list_images()
        for template in templates_in_hosts:
            if template.name not in names:
                images += [template]
        return images

    def ex_list_networks(self):
        request = '/rest/vcenter/network'
        response = self._request(request).object['value']
        networks = []
        for network in response:
            networks.append(VSphereNetwork(id=network['network'], name=network['name'], extra={'type': network['type']}))
        return networks

    def create_node(self, name, image, size=None, location=None, ex_datastore=None, ex_disks=None, ex_folder=None, ex_network=None, ex_turned_on=True):
        """
        Image can be either a vm template , a ovf template or just
        the guest OS.

        ex_folder is necessary if the image is a vm-template, this method
        will attempt to put the VM in a random folder and a warning about it
        will be issued in case the value remains `None`.
        """
        if image.extra['type'] == 'template_6_5':
            kwargs = {}
            kwargs['name'] = name
            kwargs['image'] = image
            kwargs['size'] = size
            kwargs['ex_network'] = ex_network
            kwargs['location'] = location
            for dstore in self.ex_list_datastores():
                if dstore['id'] == ex_datastore:
                    kwargs['ex_datastore'] = dstore['name']
                    break
            kwargs['folder'] = ex_folder
            if self.driver_soap is None:
                self._get_soap_driver()
            result = self.driver_soap.create_node(**kwargs)
            return result
        create_nic = False
        update_memory = False
        update_cpu = False
        create_disk = False
        update_capacity = False
        if image.extra['type'] == 'guest_OS':
            spec = {}
            spec['guest_OS'] = image.name
            spec['name'] = name
            spec['placement'] = {}
            if ex_folder is None:
                warn = 'The API(6.7) requires the folder to be given, I will place it into a random folder, after creation you might find it convenient to move it into a better folder.'
                warnings.warn(warn)
                folders = self.ex_list_folders()
                for folder in folders:
                    if folder['type'] == 'VIRTUAL_MACHINE':
                        ex_folder = folder['folder']
                if ex_folder is None:
                    msg = 'No suitable folder vor VMs found, please create one'
                    raise ProviderError(msg, 404)
            spec['placement']['folder'] = ex_folder
            if location.extra['type'] == 'host':
                spec['placement']['host'] = location.id
            elif location.extra['type'] == 'cluster':
                spec['placement']['cluster'] = location.id
            elif location.extra['type'] == 'resource_pool':
                spec['placement']['resource_pool'] = location.id
            spec['placement']['datastore'] = ex_datastore
            cpu = size.extra.get('cpu', 1)
            spec['cpu'] = {'count': cpu}
            spec['memory'] = {'size_MiB': size.ram}
            if size.disk:
                disk = {}
                disk['new_vmdk'] = {}
                disk['new_vmdk']['capacity'] = size.disk * 1024 ** 3
                spec['disks'] = [disk]
            if ex_network:
                nic = {}
                nic['mac_type'] = 'GENERATED'
                nic['backing'] = {}
                nic['backing']['type'] = 'STANDARD_PORTGROUP'
                nic['backing']['network'] = ex_network
                nic['start_connected'] = True
                spec['nics'] = [nic]
            create_request = '/rest/vcenter/vm'
            data = json.dumps({'spec': spec})
        elif image.extra['type'] == 'ovf':
            ovf_request = '/rest/com/vmware/vcenter/ovf/library-item/id:{}?~action=filter'.format(image.id)
            spec = {}
            spec['target'] = {}
            if location.extra.get('type') == 'resource-pool':
                spec['target']['resource_pool_id'] = location.id
            elif location.extra.get('type') == 'host':
                resource_pool = self._get_resource_pool(host_id=location.id)
                if not resource_pool:
                    msg = 'Could not find resource-pool for given location (host). Please make sure the location is valid.'
                    raise VSphereException(code='504', message=msg)
                spec['target']['resource_pool_id'] = resource_pool
                spec['target']['host_id'] = location.id
            elif location.extra.get('type') == 'cluster':
                resource_pool = self._get_resource_pool(cluster_id=location.id)
                if not resource_pool:
                    msg = 'Could not find resource-pool for given location (cluster). Please make sure the location is valid.'
                    raise VSphereException(code='504', message=msg)
                spec['target']['resource_pool_id'] = resource_pool
            ovf = self._request(ovf_request, method='POST', data=json.dumps(spec)).object['value']
            spec['deployment_spec'] = {}
            spec['deployment_spec']['name'] = name
            spec['deployment_spec']['accept_all_EULA'] = True
            if ex_network and ovf['networks']:
                spec['deployment_spec']['network_mappings'] = [{'key': ovf['networks'][0], 'value': ex_network}]
            elif not ovf['networks'] and ex_network:
                create_nic = True
            if ex_datastore:
                spec['deployment_spec']['storage_mappings'] = []
                store_map = {'type': 'DATASTORE', 'datastore_id': ex_datastore}
                spec['deployment_spec']['storage_mappings'].append(store_map)
            if size and size.ram:
                update_memory = True
            if size and size.extra and size.extra.get('cpu'):
                update_cpu = True
            if size and size.disk:
                pass
            if ex_disks:
                create_disk = True
            create_request = '/rest/com/vmware/vcenter/ovf/library-item/id:{}?~action=deploy'.format(image.id)
            data = json.dumps({'target': spec['target'], 'deployment_spec': spec['deployment_spec']})
        elif image.extra['type'] == 'vm-template':
            tp_request = '/rest/vcenter/vm-template/library-items/' + image.id
            template = self._request(tp_request).object['value']
            spec = {}
            spec['name'] = name
            if ex_datastore:
                spec['disk_storage'] = {}
                spec['disk_storage']['datastore'] = ex_datastore
            spec['placement'] = {}
            if not ex_folder:
                warn = 'The API(6.7) requires the folder to be given, I will place it into a random folder, after creation you might find it convenient to move it into a better folder.'
                warnings.warn(warn)
                folders = self.ex_list_folders()
                for folder in folders:
                    if folder['type'] == 'VIRTUAL_MACHINE':
                        ex_folder = folder['folder']
                if ex_folder is None:
                    msg = 'No suitable folder vor VMs found, please create one'
                    raise ProviderError(msg, 404)
            spec['placement']['folder'] = ex_folder
            if location.extra['type'] == 'host':
                spec['placement']['host'] = location.id
            elif location.extra['type'] == 'cluster':
                spec['placement']['cluster'] = location.id
            spec['hardware_customization'] = {}
            if ex_network:
                nics = template['nics']
                if len(nics) > 0:
                    nic = nics[0]
                    spec['hardware_customization']['nics'] = [{'key': nic['key'], 'value': {'network': ex_network}}]
                else:
                    create_nic = True
            spec['powered_on'] = False
            if size:
                if size.ram:
                    spec['hardware_customization']['memory_update'] = {'memory': int(size.ram)}
                if size.extra.get('cpu'):
                    spec['hardware_customization']['cpu_update'] = {'num_cpus': size.extra['cpu']}
                if size.disk:
                    if not len(template['disks']) > 0:
                        create_disk = True
                    else:
                        capacity = size.disk * 1024 * 1024 * 1024
                        dsk = template['disks'][0]['key']
                        if template['disks'][0]['value']['capacity'] < capacity:
                            update = {'capacity': capacity}
                            spec['hardware_customization']['disks_to_update'] = [{'key': dsk, 'value': update}]
            create_request = '/rest/vcenter/vm-template/library-items/{}/?action=deploy'.format(image.id)
            data = json.dumps({'spec': spec})
        result = self._request(create_request, method='POST', data=data)
        node_id = result.object['value']
        if image.extra['type'] == 'ovf':
            node_id = node_id['resource_id']['id']
        node = self.list_nodes(ex_filter_vms=node_id)[0]
        if create_nic:
            self.ex_add_nic(node, ex_network)
        if update_memory:
            self.ex_update_memory(node, size.ram)
        if update_cpu:
            self.ex_update_cpu(node, size.extra['cpu'])
        if create_disk:
            pass
        if update_capacity:
            pass
        if ex_turned_on:
            self.start_node(node)
        return node

    def ex_list_snapshots(self, node):
        """
        List node snapshots
        """
        if self.driver_soap is None:
            self._get_soap_driver()
        return self.driver_soap.ex_list_snapshots(node)

    def ex_create_snapshot(self, node, snapshot_name, description='', dump_memory=False, quiesce=False):
        """
        Create node snapshot
        """
        if self.driver_soap is None:
            self._get_soap_driver()
        return self.driver_soap.ex_create_snapshot(node, snapshot_name, description=description, dump_memory=dump_memory, quiesce=False)

    def ex_remove_snapshot(self, node, snapshot_name=None, remove_children=True):
        """
        Remove a snapshot from node.
        If snapshot_name is not defined remove the last one.
        """
        if self.driver_soap is None:
            self._get_soap_driver()
        return self.driver_soap.ex_remove_snapshot(node, snapshot_name=snapshot_name, remove_children=remove_children)

    def ex_revert_to_snapshot(self, node, snapshot_name=None):
        """
        Revert node to a specific snapshot.
        If snapshot_name is not defined revert to the last one.
        """
        if self.driver_soap is None:
            self._get_soap_driver()
        return self.driver_soap.ex_revert_to_snapshot(node, snapshot_name=snapshot_name)

    def ex_open_console(self, vm_id):
        if self.driver_soap is None:
            self._get_soap_driver()
        return self.driver_soap.ex_open_console(vm_id)