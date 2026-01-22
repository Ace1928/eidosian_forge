import os
import re
import binascii
import itertools
from copy import copy
from datetime import datetime
from libcloud.utils.py3 import httplib
from libcloud.compute.base import (
from libcloud.common.linode import (
from libcloud.compute.types import Provider, NodeState, StorageVolumeState
from libcloud.utils.networking import is_private_subnet
class LinodeNodeDriverV3(LinodeNodeDriver):
    """libcloud driver for the Linode API

    Rough mapping of which is which:

    - list_nodes              linode.list
    - reboot_node             linode.reboot
    - destroy_node            linode.delete
    - create_node             linode.create, linode.update,
                              linode.disk.createfromdistribution,
                              linode.disk.create, linode.config.create,
                              linode.ip.addprivate, linode.boot
    - list_sizes              avail.linodeplans
    - list_images             avail.distributions
    - list_locations          avail.datacenters
    - list_volumes            linode.disk.list
    - destroy_volume          linode.disk.delete

    For more information on the Linode API, be sure to read the reference:

        http://www.linode.com/api/
    """
    connectionCls = LinodeConnection
    _linode_plan_ids = LINODE_PLAN_IDS
    _linode_disk_filesystems = LINODE_DISK_FILESYSTEMS
    features = {'create_node': ['ssh_key', 'password']}

    def __init__(self, key, secret=None, secure=True, host=None, port=None, api_version=None, region=None, **kwargs):
        """Instantiate the driver with the given API key

        :param   key: the API key to use (required)
        :type    key: ``str``

        :rtype: ``None``
        """
        self.datacenter = None
        NodeDriver.__init__(self, key)
    LINODE_STATES = {-2: NodeState.UNKNOWN, -1: NodeState.PENDING, 0: NodeState.PENDING, 1: NodeState.RUNNING, 2: NodeState.STOPPED, 3: NodeState.REBOOTING, 4: NodeState.UNKNOWN}

    def list_nodes(self):
        """
        List all Linodes that the API key can access

        This call will return all Linodes that the API key in use has access
         to.
        If a node is in this list, rebooting will work; however, creation and
        destruction are a separate grant.

        :return: List of node objects that the API key can access
        :rtype: ``list`` of :class:`Node`
        """
        params = {'api_action': 'linode.list'}
        data = self.connection.request(API_ROOT, params=params).objects[0]
        return self._to_nodes(data)

    def start_node(self, node):
        """
        Boot the given Linode

        """
        params = {'api_action': 'linode.boot', 'LinodeID': node.id}
        self.connection.request(API_ROOT, params=params)
        return True

    def stop_node(self, node):
        """
        Shutdown the given Linode

        """
        params = {'api_action': 'linode.shutdown', 'LinodeID': node.id}
        self.connection.request(API_ROOT, params=params)
        return True

    def reboot_node(self, node):
        """
        Reboot the given Linode

        Will issue a shutdown job followed by a boot job, using the last booted
        configuration.  In most cases, this will be the only configuration.

        :param      node: the Linode to reboot
        :type       node: :class:`Node`

        :rtype: ``bool``
        """
        params = {'api_action': 'linode.reboot', 'LinodeID': node.id}
        self.connection.request(API_ROOT, params=params)
        return True

    def destroy_node(self, node):
        """Destroy the given Linode

        Will remove the Linode from the account and issue a prorated credit. A
        grant for removing Linodes from the account is required, otherwise this
        method will fail.

        In most cases, all disk images must be removed from a Linode before the
        Linode can be removed; however, this call explicitly skips those
        safeguards. There is no going back from this method.

        :param       node: the Linode to destroy
        :type        node: :class:`Node`

        :rtype: ``bool``
        """
        params = {'api_action': 'linode.delete', 'LinodeID': node.id, 'skipChecks': True}
        self.connection.request(API_ROOT, params=params)
        return True

    def create_node(self, name, image, size, auth, location=None, ex_swap=None, ex_rsize=None, ex_kernel=None, ex_payment=None, ex_comment=None, ex_private=False, lconfig=None, lroot=None, lswap=None):
        """Create a new Linode, deploy a Linux distribution, and boot

        This call abstracts much of the functionality of provisioning a Linode
        and getting it booted.  A global grant to add Linodes to the account is
        required, as this call will result in a billing charge.

        Note that there is a safety valve of 5 Linodes per hour, in order to
        prevent a runaway script from ruining your day.

        :keyword name: the name to assign the Linode (mandatory)
        :type    name: ``str``

        :keyword image: which distribution to deploy on the Linode (mandatory)
        :type    image: :class:`NodeImage`

        :keyword size: the plan size to create (mandatory)
        :type    size: :class:`NodeSize`

        :keyword auth: an SSH key or root password (mandatory)
        :type    auth: :class:`NodeAuthSSHKey` or :class:`NodeAuthPassword`

        :keyword location: which datacenter to create the Linode in
        :type    location: :class:`NodeLocation`

        :keyword ex_swap: size of the swap partition in MB (128)
        :type    ex_swap: ``int``

        :keyword ex_rsize: size of the root partition in MB (plan size - swap).
        :type    ex_rsize: ``int``

        :keyword ex_kernel: a kernel ID from avail.kernels (Latest 2.6 Stable).
        :type    ex_kernel: ``str``

        :keyword ex_payment: one of 1, 12, or 24; subscription length (1)
        :type    ex_payment: ``int``

        :keyword ex_comment: a small comment for the configuration (libcloud)
        :type    ex_comment: ``str``

        :keyword ex_private: whether or not to request a private IP (False)
        :type    ex_private: ``bool``

        :keyword lconfig: what to call the configuration (generated)
        :type    lconfig: ``str``

        :keyword lroot: what to call the root image (generated)
        :type    lroot: ``str``

        :keyword lswap: what to call the swap space (generated)
        :type    lswap: ``str``

        :return: Node representing the newly-created Linode
        :rtype: :class:`Node`
        """
        auth = self._get_and_check_auth(auth)
        if location:
            chosen = location.id
        elif self.datacenter:
            chosen = self.datacenter
        else:
            raise LinodeException(251, 'Need to select a datacenter first')
        plans = self.list_sizes()
        if size.id not in [p.id for p in plans]:
            raise LinodeException(251, 'Invalid plan ID -- avail.plans')
        payment = '1' if not ex_payment else str(ex_payment)
        if payment not in ['1', '12', '24']:
            raise LinodeException(251, 'Invalid subscription (1, 12, 24)')
        ssh = None
        root = None
        if isinstance(auth, NodeAuthSSHKey):
            ssh = auth.pubkey
        elif isinstance(auth, NodeAuthPassword):
            root = auth.password
        if not ssh and (not root):
            raise LinodeException(251, 'Need SSH key or root password')
        if root is not None and len(root) < 6:
            raise LinodeException(251, 'Root password is too short')
        try:
            swap = 128 if not ex_swap else int(ex_swap)
        except Exception:
            raise LinodeException(251, 'Need an integer swap size')
        imagesize = size.disk - swap if not ex_rsize else int(ex_rsize)
        if imagesize + swap > size.disk:
            raise LinodeException(251, 'Total disk images are too big')
        distros = self.list_images()
        if image.id not in [d.id for d in distros]:
            raise LinodeException(251, 'Invalid distro -- avail.distributions')
        if ex_kernel:
            kernel = ex_kernel
        elif image.extra['64bit']:
            kernel = 138
        else:
            kernel = 137
        params = {'api_action': 'avail.kernels'}
        kernels = self.connection.request(API_ROOT, params=params).objects[0]
        if kernel not in [z['KERNELID'] for z in kernels]:
            raise LinodeException(251, 'Invalid kernel -- avail.kernels')
        comments = 'Created by Apache libcloud <https://www.libcloud.org>' if not ex_comment else ex_comment
        params = {'api_action': 'linode.create', 'DatacenterID': chosen, 'PlanID': size.id, 'PaymentTerm': payment}
        data = self.connection.request(API_ROOT, params=params).objects[0]
        linode = {'id': data['LinodeID']}
        params = {'api_action': 'linode.update', 'LinodeID': linode['id'], 'Label': name}
        self.connection.request(API_ROOT, params=params)
        if ex_private:
            params = {'api_action': 'linode.ip.addprivate', 'LinodeID': linode['id']}
            self.connection.request(API_ROOT, params=params)
        label = {'lconfig': '[%s] Configuration Profile' % linode['id'], 'lroot': '[{}] {} Disk Image'.format(linode['id'], image.name), 'lswap': '[%s] Swap Space' % linode['id']}
        if lconfig:
            label['lconfig'] = lconfig
        if lroot:
            label['lroot'] = lroot
        if lswap:
            label['lswap'] = lswap
        if not root:
            root = binascii.b2a_base64(os.urandom(8)).decode('ascii').strip()
        params = {'api_action': 'linode.disk.createfromdistribution', 'LinodeID': linode['id'], 'DistributionID': image.id, 'Label': label['lroot'], 'Size': imagesize, 'rootPass': root}
        if ssh:
            params['rootSSHKey'] = ssh
        data = self.connection.request(API_ROOT, params=params).objects[0]
        linode['rootimage'] = data['DiskID']
        params = {'api_action': 'linode.disk.create', 'LinodeID': linode['id'], 'Label': label['lswap'], 'Type': 'swap', 'Size': swap}
        data = self.connection.request(API_ROOT, params=params).objects[0]
        linode['swapimage'] = data['DiskID']
        disks = '{},{},,,,,,,'.format(linode['rootimage'], linode['swapimage'])
        params = {'api_action': 'linode.config.create', 'LinodeID': linode['id'], 'KernelID': kernel, 'Label': label['lconfig'], 'Comments': comments, 'DiskList': disks}
        if ex_private:
            params['helper_network'] = True
            params['helper_distro'] = True
        data = self.connection.request(API_ROOT, params=params).objects[0]
        linode['config'] = data['ConfigID']
        params = {'api_action': 'linode.boot', 'LinodeID': linode['id'], 'ConfigID': linode['config']}
        self.connection.request(API_ROOT, params=params)
        params = {'api_action': 'linode.list', 'LinodeID': linode['id']}
        data = self.connection.request(API_ROOT, params=params).objects[0]
        nodes = self._to_nodes(data)
        if len(nodes) == 1:
            node = nodes[0]
            if getattr(auth, 'generated', False):
                node.extra['password'] = auth.password
            return node
        return None

    def ex_resize_node(self, node, size):
        """Resizes a Linode from one plan to another

        Immediately shuts the Linode down, charges/credits the account,
        and issue a migration to another host server.
        Requires a size (numeric), which is the desired PlanID available from
        avail.LinodePlans()
        After resize is complete the node needs to be booted
        """
        params = {'api_action': 'linode.resize', 'LinodeID': node.id, 'PlanID': size}
        self.connection.request(API_ROOT, params=params)
        return True

    def ex_start_node(self, node):
        return self.start_node(node=node)

    def ex_stop_node(self, node):
        return self.stop_node(node=node)

    def ex_rename_node(self, node, name):
        """Renames a node"""
        params = {'api_action': 'linode.update', 'LinodeID': node.id, 'Label': name}
        self.connection.request(API_ROOT, params=params)
        return True

    def list_sizes(self, location=None):
        """
        List available Linode plans

        Gets the sizes that can be used for creating a Linode.  Since available
        Linode plans vary per-location, this method can also be passed a
        location to filter the availability.

        :keyword location: the facility to retrieve plans in
        :type    location: :class:`NodeLocation`

        :rtype: ``list`` of :class:`NodeSize`
        """
        params = {'api_action': 'avail.linodeplans'}
        data = self.connection.request(API_ROOT, params=params).objects[0]
        sizes = []
        for obj in data:
            n = NodeSize(id=obj['PLANID'], name=obj['LABEL'], ram=obj['RAM'], disk=obj['DISK'] * 1024, bandwidth=obj['XFER'], price=obj['PRICE'], driver=self.connection.driver)
            sizes.append(n)
        return sizes

    def list_images(self):
        """
        List available Linux distributions

        Retrieve all Linux distributions that can be deployed to a Linode.

        :rtype: ``list`` of :class:`NodeImage`
        """
        params = {'api_action': 'avail.distributions'}
        data = self.connection.request(API_ROOT, params=params).objects[0]
        distros = []
        for obj in data:
            i = NodeImage(id=obj['DISTRIBUTIONID'], name=obj['LABEL'], driver=self.connection.driver, extra={'pvops': obj['REQUIRESPVOPSKERNEL'], '64bit': obj['IS64BIT']})
            distros.append(i)
        return distros

    def list_locations(self):
        """
        List available facilities for deployment

        Retrieve all facilities that a Linode can be deployed in.

        :rtype: ``list`` of :class:`NodeLocation`
        """
        params = {'api_action': 'avail.datacenters'}
        data = self.connection.request(API_ROOT, params=params).objects[0]
        nl = []
        for dc in data:
            country = None
            if 'USA' in dc['LOCATION']:
                country = 'US'
            elif 'UK' in dc['LOCATION']:
                country = 'GB'
            elif 'JP' in dc['LOCATION']:
                country = 'JP'
            else:
                country = '??'
            nl.append(NodeLocation(dc['DATACENTERID'], dc['LOCATION'], country, self))
        return nl

    def linode_set_datacenter(self, dc):
        """
        Set the default datacenter for Linode creation

        Since Linodes must be created in a facility, this function sets the
        default that :class:`create_node` will use.  If a location keyword is
        not passed to :class:`create_node`, this method must have already been
        used.

        :keyword dc: the datacenter to create Linodes in unless specified
        :type    dc: :class:`NodeLocation`

        :rtype: ``bool``
        """
        did = dc.id
        params = {'api_action': 'avail.datacenters'}
        data = self.connection.request(API_ROOT, params=params).objects[0]
        for datacenter in data:
            if did == dc['DATACENTERID']:
                self.datacenter = did
                return
        dcs = ', '.join([d['DATACENTERID'] for d in data])
        self.datacenter = None
        raise LinodeException(253, 'Invalid datacenter (use one of %s)' % dcs)

    def destroy_volume(self, volume):
        """
        Destroys disk volume for the Linode. Linode id is to be provided as
        extra["LinodeId"] within :class:`StorageVolume`. It can be retrieved
        by :meth:`libcloud.compute.drivers.linode.LinodeNodeDriver                 .ex_list_volumes`.

        :param volume: Volume to be destroyed
        :type volume: :class:`StorageVolume`

        :rtype: ``bool``
        """
        if not isinstance(volume, StorageVolume):
            raise LinodeException(253, 'Invalid volume instance')
        if volume.extra['LINODEID'] is None:
            raise LinodeException(253, 'Missing LinodeID')
        params = {'api_action': 'linode.disk.delete', 'LinodeID': volume.extra['LINODEID'], 'DiskID': volume.id}
        self.connection.request(API_ROOT, params=params)
        return True

    def ex_create_volume(self, size, name, node, fs_type):
        """
        Create disk for the Linode.

        :keyword    size: Size of volume in megabytes (required)
        :type       size: ``int``

        :keyword    name: Name of the volume to be created
        :type       name: ``str``

        :keyword    node: Node to attach volume to.
        :type       node: :class:`Node`

        :keyword    fs_type: The formatted type of this disk. Valid types are:
                             ext3, ext4, swap, raw
        :type       fs_type: ``str``


        :return: StorageVolume representing the newly-created volume
        :rtype: :class:`StorageVolume`
        """
        if not isinstance(node, Node):
            raise LinodeException(253, 'Invalid node instance')
        total_space = node.extra['TOTALHD']
        existing_volumes = self.ex_list_volumes(node)
        used_space = 0
        for volume in existing_volumes:
            used_space = used_space + volume.size
        available_space = total_space - used_space
        if available_space < size:
            raise LinodeException(253, 'Volume size too big. Available space                    %d' % available_space)
        if fs_type not in self._linode_disk_filesystems:
            raise LinodeException(253, 'Not valid filesystem type')
        params = {'api_action': 'linode.disk.create', 'LinodeID': node.id, 'Label': name, 'Type': fs_type, 'Size': size}
        data = self.connection.request(API_ROOT, params=params).objects[0]
        volume = data['DiskID']
        params = {'api_action': 'linode.disk.list', 'LinodeID': node.id, 'DiskID': volume}
        data = self.connection.request(API_ROOT, params=params).objects[0]
        return self._to_volumes(data)[0]

    def ex_list_volumes(self, node, disk_id=None):
        """
        List existing disk volumes for for given Linode.

        :keyword    node: Node to list disk volumes for. (required)
        :type       node: :class:`Node`

        :keyword    disk_id: Id for specific disk volume. (optional)
        :type       disk_id: ``int``

        :rtype: ``list`` of :class:`StorageVolume`
        """
        if not isinstance(node, Node):
            raise LinodeException(253, 'Invalid node instance')
        params = {'api_action': 'linode.disk.list', 'LinodeID': node.id}
        if disk_id is not None:
            params['DiskID'] = disk_id
        data = self.connection.request(API_ROOT, params=params).objects[0]
        return self._to_volumes(data)

    def _to_volumes(self, objs):
        """
        Convert returned JSON volumes into StorageVolume instances

        :keyword    objs: ``list`` of JSON dictionaries representing the
                         StorageVolumes
        :type       objs: ``list``

        :return: ``list`` of :class:`StorageVolume`s
        """
        volumes = {}
        for o in objs:
            vid = o['DISKID']
            volumes[vid] = vol = StorageVolume(id=vid, name=o['LABEL'], size=int(o['SIZE']), driver=self.connection.driver)
            vol.extra = copy(o)
        return list(volumes.values())

    def _to_nodes(self, objs):
        """Convert returned JSON Linodes into Node instances

        :keyword objs: ``list`` of JSON dictionaries representing the Linodes
        :type objs: ``list``
        :return: ``list`` of :class:`Node`s"""
        nodes = {}
        batch = []
        for o in objs:
            lid = o['LINODEID']
            nodes[lid] = n = Node(id=lid, name=o['LABEL'], public_ips=[], private_ips=[], state=self.LINODE_STATES[o['STATUS']], driver=self.connection.driver)
            n.extra = copy(o)
            n.extra['PLANID'] = self._linode_plan_ids.get(o.get('TOTALRAM'))
            batch.append({'api_action': 'linode.ip.list', 'LinodeID': lid})
        ip_answers = []
        args = [iter(batch)] * 25
        for twenty_five in itertools.zip_longest(*args):
            twenty_five = [q for q in twenty_five if q]
            params = {'api_action': 'batch', 'api_requestArray': json.dumps(twenty_five)}
            req = self.connection.request(API_ROOT, params=params)
            if not req.success() or len(req.objects) == 0:
                return None
            ip_answers.extend(req.objects)
        for ip_list in ip_answers:
            for ip in ip_list:
                lid = ip['LINODEID']
                which = nodes[lid].public_ips if ip['ISPUBLIC'] == 1 else nodes[lid].private_ips
                which.append(ip['IPADDRESS'])
        return list(nodes.values())