import json
import time
import base64
from typing import Any, Dict, List, Union, Optional
from functools import update_wrapper
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, InvalidCredsError, ServiceUnavailableError
from libcloud.common.vultr import (
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, StorageVolumeState, VolumeSnapshotState
from libcloud.utils.iso8601 import parse_date
from libcloud.utils.publickey import get_pubkey_openssh_fingerprint
class VultrNodeDriverV2(VultrNodeDriver):
    """
    Vultr API v2 NodeDriver.
    """
    connectionCls = VultrConnectionV2
    NODE_STATE_MAP = {'active': NodeState.RUNNING, 'halted': NodeState.STOPPED, 'rebooting': NodeState.REBOOTING, 'resizing': NodeState.RECONFIGURING, 'pending': NodeState.PENDING}
    VOLUME_STATE_MAP = {'active': StorageVolumeState.AVAILABLE, 'pending': StorageVolumeState.CREATING}
    SNAPSHOT_STATE_MAP = {'complete': VolumeSnapshotState.AVAILABLE, 'pending': VolumeSnapshotState.CREATING}

    def list_nodes(self, ex_list_bare_metals: bool=True) -> List[Node]:
        """List all nodes.

        :keyword ex_list_bare_metals: Whether to fetch bare metal nodes.
        :type    ex_list_bare_metals: ``bool``

        :return:  list of node objects
        :rtype: ``list`` of :class: `Node`
        """
        data = self._paginated_request('/v2/instances', 'instances')
        nodes = [self._to_node(item) for item in data]
        if ex_list_bare_metals:
            nodes += self.ex_list_bare_metal_nodes()
        return nodes

    def create_node(self, name: str, size: NodeSize, location: NodeLocation, image: Optional[NodeImage]=None, ex_ssh_key_ids: Optional[List[str]]=None, ex_private_network_ids: Optional[List[str]]=None, ex_snapshot: Union[VultrNodeSnapshot, str, None]=None, ex_enable_ipv6: bool=False, ex_backups: bool=False, ex_userdata: Optional[str]=None, ex_ddos_protection: bool=False, ex_enable_private_network: bool=False, ex_ipxe_chain_url: Optional[str]=None, ex_iso_id: Optional[str]=None, ex_script_id: Optional[str]=None, ex_image_id: Optional[str]=None, ex_activation_email: bool=False, ex_hostname: Optional[str]=None, ex_tag: Optional[str]=None, ex_firewall_group_id: Optional[str]=None, ex_reserved_ipv4: Optional[str]=None, ex_persistent_pxe: bool=False) -> Node:
        """Create a new node.

        :param name: The new node's name.
        :type name: ``str``

        :param size: The size to use to create the node.
        :type size: :class: `NodeSize`

        :param location: The location to provision the node.
        :type location: :class: `NodeLocation`

        :keyword image:  The image to use to provision the node.
        :type    image:  :class: `NodeImage`

        :keyword ex_ssh_key_ids: List of SSH keys to install on this node.
        :type    ex_ssh_key_ids: ``list`` of ``str``

        :keyword ex_private_network_ids: The network ids to attach to node.
                                         This parameter takes precedence over
                                         ex_enable_private_network (VPS only)
        :type    ex_private_network_ids: ``list`` of ``str``

        :keyword ex_snapshot: The snapshot to use when deploying the node.
                                 Mutually exclusive with image,
        :type    ex_snapshot: :class: `VultrNodeSnapshot` or ``str``

        :keyword ex_enable_ipv6: Whether to enable IPv6.
        :type    ex_enable_ipv6: ``bool``

        :keyword ex_backups: Enable automatic backups for the node. (VPS only)
        :type    ex_backups: ``bool``

        :keyword ex_userdata: String containing user data
        :type    ex_userdata: ``str``

        :keyword ex_ddos_protection: Enable DDoS protection (VPS only)
        :type    ex_ddos_protection: ``bool``

        :keyword ex_enable_private_network: Enable private networking.
                                            Mutually exclusive with
                                             ex_private_network_ids.
                                            (VPS only)
        :type    ex_enable_private_network: ``bool``

        :keyword ex_ipxe_chain_url: The URL location of the iPXE chainloader
                                    (VPS only)
        :type    ex_ipxe_chain_url: ``str``

        :keyword ex_iso_id: The ISO id to use when deploying this node.
                            (VPS only)
        :type    ex_iso_id: ``str``

        :keyword ex_script_id: The startup script id to use when deploying
                               this node.
        :type    ex_script_id: ``str``

        :keyword ex_image_id: The Application image_id to use when deploying
                              this node.
        :type    ex_image_id: ``str``

        :keyword ex_activation_email: Notify by email after deployment.
        :type    ex_activation_email: ``bool``

        :keyword ex_hostname: The hostname to use when deploying this node.
        :type    ex_hostname: ``str``

        :keyword ex_tag: The user-supplied tag.
        :type    ex_tag: ``str``

        :keyword ex_firewall_group_id: The Firewall Group id to attach to
                                       this node. (VPS only)
        :type    ex_firewall_group_id: ``str``

        :keyword ex_reserved_ipv4: Id of the floating IP to use as the
                                   main IP of this node.
        :type    ex_reserved_ipv4: ``str``

        :keyword ex_persistent_pxe: Enable persistent PXE (Bare Metal only)
        :type    ex_persistent_pxe: ``bool``
        """
        data = {'label': name, 'region': location.id, 'plan': size.id, 'enable_ipv6': ex_enable_ipv6, 'activation_email': ex_activation_email}
        if image:
            data['os_id'] = image.id
        if ex_ssh_key_ids:
            data['sshkey_id'] = ex_ssh_key_ids
        if ex_snapshot:
            try:
                data['snapshot_id'] = ex_snapshot.id
            except AttributeError:
                data['snapshot_id'] = ex_snapshot
        if ex_userdata:
            data['user_data'] = base64.b64encode(bytes(ex_userdata, 'utf-8')).decode('utf-8')
        if ex_script_id:
            data['script_id'] = ex_script_id
        if ex_image_id:
            data['image_id'] = ex_image_id
        if ex_hostname:
            data['hostname'] = ex_hostname
        if ex_reserved_ipv4:
            data['reserved_ipv4'] = ex_reserved_ipv4
        if ex_tag:
            data['tag'] = ex_tag
        if self._is_bare_metal(size):
            if ex_persistent_pxe:
                data['persistent_pxe'] = ex_persistent_pxe
            resp = self.connection.request('/v2/bare-metals', data=json.dumps(data), method='POST')
            return self._to_node(resp.object['bare_metal'])
        else:
            if ex_private_network_ids:
                data['attach_private_network'] = ex_private_network_ids
            if ex_enable_private_network:
                data['enable_private_network'] = ex_enable_private_network
            if ex_ipxe_chain_url:
                data['ipxe_chain_url'] = ex_ipxe_chain_url
            if ex_iso_id:
                data['iso_id'] = ex_iso_id
            if ex_ddos_protection:
                data['ddos_protection'] = ex_ddos_protection
            if ex_firewall_group_id:
                data['firewall_group_id'] = ex_firewall_group_id
            if ex_backups:
                data['backups'] = 'enabled' if ex_backups is True else 'disabled'
            resp = self.connection.request('/v2/instances', data=json.dumps(data), method='POST')
            return self._to_node(resp.object['instance'])

    def reboot_node(self, node: Node) -> bool:
        """Reboot the given node.

        :param node: The node to be rebooted.
        :type node: :class: `Node`

        :rtype: ``bool``
        """
        if self._is_bare_metal(node.size):
            return self.ex_reboot_bare_metal_node(node)
        resp = self.connection.request('/v2/instances/%s/reboot' % node.id, method='POST')
        return resp.success()

    def start_node(self, node: Node) -> bool:
        """Start the given node.

        :param node: The node to be started.
        :type node: :class: `Node`

        :rtype: ``bool``
        """
        if self._is_bare_metal(node.size):
            return self.ex_start_bare_metal_node(node)
        resp = self.connection.request('/v2/instances/%s/start' % node.id, method='POST')
        return resp.success()

    def stop_node(self, node: Node) -> bool:
        """Stop the given node.

        :param node: The node to be stopped.
        :type node: :class: `Node`

        :rtype: ``bool``
        """
        if self._is_bare_metal(node.size):
            return self.ex_stop_bare_metal_node(node)
        return self.ex_stop_nodes([node])

    def destroy_node(self, node: Node) -> bool:
        """Destroy the given node.

        :param node: The node to be destroyed.
        :type node: :class: `Node`

        :rtype: ``bool``
        """
        if self._is_bare_metal(node.size):
            return self.ex_destroy_bare_metal_node(node)
        resp = self.connection.request('/v2/instances/%s' % node.id, method='DELETE')
        return resp.success()

    def list_sizes(self, ex_list_bare_metals: bool=True) -> List[NodeSize]:
        """List available node sizes.

        :keyword ex_list_bare_metals: Whether to fetch bare metal sizes.
        :type    ex_list_bare_metals: ``bool``

        :rtype: ``list`` of :class: `NodeSize`
        """
        data = self._paginated_request('/v2/plans', 'plans')
        sizes = [self._to_size(item) for item in data]
        if ex_list_bare_metals:
            sizes += self.ex_list_bare_metal_sizes()
        return sizes

    def list_images(self) -> List[NodeImage]:
        """List available node images.

        :rtype: ``list`` of :class: `NodeImage`
        """
        data = self._paginated_request('/v2/os', 'os')
        return [self._to_image(item) for item in data]

    def list_locations(self) -> List[NodeLocation]:
        """List available node locations.

        :rtype: ``list`` of :class: `NodeLocation`
        """
        data = self._paginated_request('/v2/regions', 'regions')
        return [self._to_location(item) for item in data]

    def list_volumes(self) -> List[StorageVolume]:
        """List storage volumes.

        :rtype: ``list`` of :class:`StorageVolume`
        """
        data = self._paginated_request('/v2/blocks', 'blocks')
        return [self._to_volume(item) for item in data]

    def create_volume(self, size: int, name: str, location: Union[NodeLocation, str]) -> StorageVolume:
        """Create a new volume.

        :param size: Size of the volume in gigabytes.        Size may range between 10 and 10000.
        :type size: ``int``

        :param name: Name of the volume to be created.
        :type name: ``str``

        :param location: Which data center to create the volume in.
        :type location: :class:`NodeLocation` or ``str``

        :return: The newly created volume.
        :rtype: :class:`StorageVolume`
        """
        data = {'label': name, 'size_gb': size}
        try:
            data['region'] = location.id
        except AttributeError:
            data['region'] = location
        resp = self.connection.request('/v2/blocks', data=json.dumps(data), method='POST')
        return self._to_volume(resp.object['block'])

    def attach_volume(self, node: Node, volume: StorageVolume, ex_live: bool=True) -> bool:
        """Attaches volume to node.

        :param node: Node to attach volume to.
        :type node: :class:`Node`

        :param volume: Volume to attach.
        :type volume: :class:`StorageVolume`

        :param ex_live: Attach the volume without restarting the node.
        :type ex_live: ``bool``

        :rytpe: ``bool``
        """
        data = {'instance_id': node.id, 'live': ex_live}
        resp = self.connection.request('/v2/blocks/%s/attach' % volume.id, data=json.dumps(data), method='POST')
        return resp.success()

    def detach_volume(self, volume: StorageVolume, ex_live: bool=True) -> bool:
        """Detaches a volume from a node.

        :param volume: Volume to be detached
        :type volume: :class:`StorageVolume`

        :param ex_live: Detach the volume without restarting the node.
        :type ex_live: ``bool``

        :rtype: ``bool``
        """
        data = {'live': ex_live}
        resp = self.connection.request('/v2/blocks/%s/detach' % volume.id, data=json.dumps(data), method='POST')
        return resp.success()

    def destroy_volume(self, volume: StorageVolume) -> bool:
        """Destroys a storage volume.

        :param volume: Volume to be destroyed
        :type  volume: :class:`StorageVolume`

        :rtype: ``bool``
        """
        resp = self.connection.request('/v2/blocks/%s' % volume.id, method='DELETE')
        return resp.success()

    def list_key_pairs(self) -> List[KeyPair]:
        """List all the available SSH key pair objects.

        :rtype: ``list`` of :class:`KeyPair`
        """
        data = self._paginated_request('/v2/ssh-keys', 'ssh_keys')
        return [self._to_key_pair(item) for item in data]

    def get_key_pair(self, key_id: str) -> KeyPair:
        """Retrieve a single key pair.

        :param key_id: ID of the key pair to retrieve.
        :type  key_id: ``str``

        :rtype: :class: `KeyPair`
        """
        resp = self.connection.request('/v2/ssh-keys/%s' % key_id)
        return self._to_key_pair(resp.object['ssh_key'])

    def import_key_pair_from_string(self, name: str, key_material: str) -> KeyPair:
        """Import a new public key from string.

        :param name: Key pair name.
        :type  name: ``str``

        :param key_material: Public key material.
        :type  key_material: ``str``

        :rtype: :class: `KeyPair`
        """
        data = {'name': name, 'ssh_key': key_material}
        resp = self.connection.request('/v2/ssh-keys', data=json.dumps(data), method='POST')
        return self._to_key_pair(resp.object['ssh_key'])

    def delete_key_pair(self, key_pair: KeyPair) -> bool:
        """Delete existing key pair.

        :param key_pair: The key pair object to delete.
        :type key_pair: :class:`.KeyPair`

        :rtype: ``bool``
        """
        resp = self.connection.request('/v2/ssh-keys/%s' % key_pair.extra['id'], method='DELETE')
        return resp.success()

    def ex_list_bare_metal_nodes(self) -> List[Node]:
        """List all bare metal nodes.

        :return:  list of node objects
        :rtype: ``list`` of :class: `Node`
        """
        data = self._paginated_request('/v2/bare-metals', 'bare_metals')
        return [self._to_node(item) for item in data]

    def ex_reboot_bare_metal_node(self, node: Node) -> bool:
        """Reboot the given bare metal node.

        :param node: The bare metal node to be rebooted.
        :type node: :class: `Node`

        :rtype: ``bool``
        """
        resp = self.connection.request('/v2/bare-metals/%s/reboot' % node.id, method='POST')
        return resp.success()

    def ex_resize_node(self, node: Node, size: NodeSize) -> bool:
        """Change size for the given node, only applicable for VPS nodes.

        :param node: The node to be resized.
        :type node: :class: `Node`

        :param size: The new size.
        :type size: :class: `NodeSize`
        """
        data = {'plan': size.id}
        resp = self.connection.request('/v2/instances/%s' % node.id, data=json.dumps(data), method='PATCH')
        return self._to_node(resp.object['instance'])

    def ex_start_bare_metal_node(self, node: Node) -> bool:
        """Start the given bare metal node.

        :param node: The bare metal node to be started.
        :type node: :class: `Node`

        :rtype: ``bool``
        """
        resp = self.connection.request('/v2/bare-metals/%s/start' % node.id, method='POST')
        return resp.success()

    def ex_stop_bare_metal_node(self, node: Node) -> bool:
        """Stop the given bare metal node.

        :param node: The bare metal node to be stopped.
        :type node: :class: `Node`

        :rtype: ``bool``
        """
        resp = self.connection.request('/v2/bare-metals/%s/halt' % node.id, method='POST')
        return resp.success()

    def ex_destroy_bare_metal_node(self, node: Node) -> bool:
        """Destroy the given bare metal node.

        :param node: The bare metal node to be destroyed.
        :type node: :class: `Node`

        :rtype: ``bool``
        """
        resp = self.connection.request('/v2/bare-metals/%s' % node.id, method='DELETE')
        return resp.success()

    def ex_get_node(self, node_id: str) -> Node:
        """Retrieve a node object.

        :param node_id: ID of the node to retrieve.
        :type snapshot_id: ``str``

        :rtype: :class: `Node`
        """
        resp = self.connection.request('/v2/instances/%s' % node_id)
        return self._to_node(resp.object['instance'])

    def ex_stop_nodes(self, nodes: List[Node]) -> bool:
        """Stops all the nodes given.

        : param nodes: A list of the nodes to stop.
        : type node: ``list`` of: class `Node`

        : rtype: ``bool``
        """
        data = {'instance_ids': [node.id for node in nodes]}
        resp = self.connection.request('/v2/instances/halt', data=json.dumps(data), method='POST')
        return resp.success()

    def ex_list_bare_metal_sizes(self) -> List[NodeSize]:
        """List bare metal sizes.

        :rtype: ``list`` of :class: `NodeSize`
        """
        data = self._paginated_request('/v2/plans-metal', 'plans_metal')
        return [self._to_size(item) for item in data]

    def ex_list_snapshots(self) -> List[VultrNodeSnapshot]:
        """List node snapshots.

        :rtype: ``list`` of :class: `VultrNodeSnapshot`
        """
        data = self._paginated_request('/v2/snapshots', 'snapshots')
        return [self._to_snapshot(item) for item in data]

    def ex_get_snapshot(self, snapshot_id: str) -> VultrNodeSnapshot:
        """Retrieve a snapshot.

        :param snapshot_id: ID of the snapshot to retrieve.
        :type snapshot_id: ``str``

        :rtype: :class: `VultrNodeSnapshot`
        """
        resp = self.connection.request('/v2/snapshots/%s' % snapshot_id)
        return self._to_snapshot(resp.object['snapshot'])

    def ex_create_snapshot(self, node: Node, description: Optional[str]=None) -> VultrNodeSnapshot:
        """Create snapshot from a node.

        :param node: Node to create the snapshot from.
        :type node: :class: `Node`

        :keyword description: A description of the snapshot.
        :type    description: ``str``

        :rtype: :class: `VultrNodeSnapshot`
        """
        data = {'instance_id': node.id}
        if description:
            data['description'] = description
        resp = self.connection.request('/v2/snapshots', data=json.dumps(data), method='POST')
        return self._to_snapshot(resp.object['snapshot'])

    def ex_delete_snapshot(self, snapshot: VultrNodeSnapshot) -> bool:
        """Delete the given snapshot.

        :param snapshot: The snapshot to delete.
        :type node: :class:`VultrNodeSnapshot`

        :rtype: ``bool``
        """
        resp = self.connection.request('/v2/snapshots/%s' % snapshot.id, method='DELETE')
        return resp.success()

    def ex_list_networks(self) -> List[VultrNetwork]:
        """List all private networks.

        :rtype: ``list`` of :class: `VultrNetwork`
        """
        data = self._paginated_request('/v2/private-networks', 'networks')
        return [self._to_network(item) for item in data]

    def ex_create_network(self, cidr_block: str, location: Union[NodeLocation, str], description: Optional[str]=None) -> VultrNetwork:
        """Create a private network.

        :param cidr_block: The CIDR block assigned to the network.
        :type  cidr_block: ``str``

        :param location: The location to create the network.
        :type  location: :class: `NodeLocation` or ``str``

        :keyword description: A description of the private network.
        :type    description: ``str``

        :rtype: :class: `VultrNetwork`
        """
        subnet, subnet_mask = cidr_block.split('/')
        data = {'v4_subnet': subnet, 'v4_subnet_mask': int(subnet_mask)}
        try:
            data['region'] = location.id
        except AttributeError:
            data['region'] = location
        if description:
            data['description'] = description
        resp = self.connection.request('/v2/private-networks', data=json.dumps(data), method='POST')
        return self._to_network(resp.object['network'])

    def ex_get_network(self, network_id: str) -> VultrNetwork:
        """Retrieve a private network.

        :param network_id: ID of the network to retrieve.
        :type network_id: ``str``

        :rtype: :class: `VultrNetwork`
        """
        resp = self.connection.request('/v2/private-networks/%s' % network_id)
        return self._to_network(resp.object['network'])

    def ex_destroy_network(self, network: VultrNetwork) -> bool:
        """Delete a private network.

        :param network: The network to destroy.
        :type  network: :class: `VultrNetwork`

        :rtype: ``bool``
        """
        resp = self.connection.request('/v2/private-networks/%s' % network.id, method='DELETE')
        return resp.success()

    def ex_list_available_sizes_for_location(self, location: NodeLocation) -> List[str]:
        """Get a list of available sizes for the given location.

        :param location: The location to get available sizes for.
        :type location: :class: `NodeLocation`

        :return:  A list of available size IDs for the given location.
        :rtype: ``list`` of ``str``
        """
        resp = self.connection.request('/v2/regions/%s/availability' % location.id)
        return resp.object['available_plans']

    def ex_get_volume(self, volume_id: str) -> StorageVolume:
        """Retrieve a single volume.

        :param volume_id: The ID of the volume to fetch.
        :type volume_id: ``str``

        :rtype :class: `StorageVolume`
        :return: StorageVolume instance on success.
        """
        resp = self.connection.request('/v2/blocks/%s' % volume_id)
        return self._to_volume(resp.object['block'])

    def ex_resize_volume(self, volume: StorageVolume, size: int) -> bool:
        """Resize a volume.

        :param volume: The volume to resize.
        :type volume: :class:`StorageVolume`

        :param size: The new volume size in GBs.        Size may range between 10 and 10000.
        :type size: ``int``

        :rtype: ``bool``
        """
        data = {'label': volume.name, 'size_gb': size}
        resp = self.connection.request('/v2/blocks/%s' % volume.id, data=json.dumps(data), method='PATCH')
        return resp.success()

    def _is_bare_metal(self, size: Union[NodeSize, str]) -> bool:
        try:
            size_id = size.id
        except AttributeError:
            size_id = size
        return size_id.startswith('vbm')

    def _to_node(self, data: Dict[str, Any]) -> Node:
        id_ = data['id']
        name = data['label']
        public_ips = data['main_ip'].split() + data['v6_main_ip'].split()
        size = data['plan']
        image = str(data['os_id'])
        created_at = data['date_created']
        is_bare_metal = self._is_bare_metal(size)
        extra = {'location': data['region'], 'ram': data['ram'], 'disk': data['disk'], 'netmask_v4': data['netmask_v4'], 'gateway_v4': data['gateway_v4'], 'v6_network': data['v6_network'], 'v6_network_size': data['v6_network_size'], 'app_id': data['app_id'], 'image_id': data['image_id'], 'features': data['features'], 'tag': data['tag'], 'os': data['os'], 'is_bare_metal': is_bare_metal}
        if is_bare_metal:
            state = self._get_node_state(data['status'])
            extra['cpu_count'] = data['cpu_count']
            extra['mac_address'] = data['mac_address']
            private_ips = None
        else:
            state = self._get_node_state(data['status'], power_state=data['power_status'])
            extra['vcpu_count'] = data['vcpu_count']
            extra['allowed_bandwidth'] = data['allowed_bandwidth']
            extra['power_status'] = data['power_status']
            extra['server_status'] = data['server_status']
            extra['firewall_group_id'] = data['firewall_group_id']
            private_ips = data['internal_ip'].split()
        return Node(id=id_, name=name, state=state, public_ips=public_ips, private_ips=private_ips, driver=self, size=size, image=image, extra=extra, created_at=created_at)

    def _to_volume(self, data: Dict[str, Any]) -> StorageVolume:
        id_ = data['id']
        name = data['label']
        size = data['size_gb']
        try:
            state = self.VOLUME_STATE_MAP[data['status']]
        except KeyError:
            state = StorageVolumeState.UNKNOWN
        extra = {'date_created': data['date_created'], 'cost': data['cost'], 'location': data['region'], 'attached_to_instance': data['attached_to_instance'], 'mount_id': data['mount_id']}
        return StorageVolume(id=id_, name=name, size=size, driver=self, state=state, extra=extra)

    def _get_node_state(self, state: str, power_state: Optional[str]=None) -> NodeState:
        try:
            state = self.NODE_STATE_MAP[state]
        except KeyError:
            state = NodeState.UNKNOWN
        if power_state is None:
            return state
        if state == NodeState.RUNNING and power_state != 'running':
            state = NodeState.STOPPED
        return state

    def _to_key_pair(self, data: Dict[str, Any]) -> KeyPair:
        name = data['name']
        public_key = data['ssh_key']
        try:
            fingerprint = get_pubkey_openssh_fingerprint(public_key)
        except RuntimeError:
            fingerprint = None
        extra = {'id': data['id'], 'date_created': data['date_created']}
        return KeyPair(name=name, public_key=public_key, fingerprint=fingerprint, driver=self, extra=extra)

    def _to_location(self, data: Dict[str, Any]) -> NodeLocation:
        id_ = data['id']
        name = data['city']
        country = data['country']
        extra = {'continent': data['continent'], 'option': data['options']}
        return NodeLocation(id=id_, name=name, country=country, driver=self, extra=extra)

    def _to_image(self, data: Dict[str, Any]) -> NodeImage:
        id_ = data['id']
        name = data['name']
        extra = {'arch': data['arch'], 'family': data['family']}
        return NodeImage(id=id_, name=name, driver=self, extra=extra)

    def _to_size(self, data: Dict[str, Any]) -> NodeSize:
        id_ = data['id']
        ram = data['ram']
        disk = data['disk']
        bandwidth = data['bandwidth']
        price = data['monthly_cost']
        is_bare_metal = self._is_bare_metal(id_)
        extra = {'locations': data['locations'], 'type': data['type'], 'disk_count': data['disk_count'], 'is_bare_metal': is_bare_metal}
        if is_bare_metal is False:
            extra['vcpu_count'] = data['vcpu_count']
        else:
            extra['cpu_count'] = data['cpu_count']
            extra['cpu_model'] = data['cpu_model']
            extra['cpu_threads'] = data['cpu_threads']
        return NodeSize(id=id_, name=id_, ram=ram, disk=disk, bandwidth=bandwidth, price=price, driver=self, extra=extra)

    def _to_network(self, data: Dict[str, Any]) -> VultrNetwork:
        id_ = data['id']
        cidr_block = '{}/{}'.format(data['v4_subnet'], data['v4_subnet_mask'])
        location = data['region']
        extra = {'description': data['description'], 'date_created': data['date_created']}
        return VultrNetwork(id=id_, cidr_block=cidr_block, location=location, extra=extra)

    def _to_snapshot(self, data: Dict[str, Any]) -> VultrNodeSnapshot:
        id_ = data['id']
        created = data['date_created']
        size = data['size'] / 1024 / 1024 / 1024
        try:
            state = self.SNAPSHOT_STATE_MAP[data['status']]
        except KeyError:
            state = VolumeSnapshotState.UNKNOWN
        extra = {'description': data['description'], 'os_id': data['os_id'], 'app_id': data['app_id']}
        return VultrNodeSnapshot(id=id_, size=size, created=created, state=state, extra=extra, driver=self)

    def _paginated_request(self, url: str, key: str, params: Optional[Dict[str, Any]]=None) -> List[Any]:
        """Perform multiple calls to get the full list of items when
        the API responses are paginated.

        :param url: API endpoint
        :type url: ``str``

        :param key: Result object key
        :type key: ``str``

        :param params: Request parameters
        :type params: ``dict``

        :return: ``list`` of API response objects
        :rtype: ``list``
        """
        params = params if params is not None else {}
        resp = self.connection.request(url, params=params).object
        data = list(resp.get(key, []))
        objects = data
        while True:
            next_page = resp['meta']['links']['next']
            if next_page:
                params['cursor'] = next_page
                resp = self.connection.request(url, params=params).object
                data = list(resp.get(key, []))
                objects.extend(data)
            else:
                return objects