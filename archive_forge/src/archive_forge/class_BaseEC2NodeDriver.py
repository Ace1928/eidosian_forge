import re
import copy
import time
import base64
import warnings
from typing import List
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, basestring, ensure_string
from libcloud.utils.xml import findall, findattr, findtext, fixxpath
from libcloud.common.aws import DEFAULT_SIGNATURE_VERSION, AWSBaseResponse, SignedAWSConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date, parse_date_allow_empty
from libcloud.utils.publickey import get_pubkey_comment, get_pubkey_ssh2_fingerprint
from libcloud.compute.providers import Provider
from libcloud.compute.constants.ec2_region_details_partial import (
class BaseEC2NodeDriver(NodeDriver):
    """
    Base Amazon EC2 node driver.

    Used for main EC2 and other derivate driver classes to inherit from it.
    """
    connectionCls = EC2Connection
    features = {'create_node': ['ssh_key']}
    path = '/'
    region_name = ''
    country = ''
    signature_version = DEFAULT_SIGNATURE_VERSION
    NODE_STATE_MAP = {'pending': NodeState.PENDING, 'running': NodeState.RUNNING, 'shutting-down': NodeState.UNKNOWN, 'terminated': NodeState.TERMINATED}
    VOLUME_STATE_MAP = {'available': StorageVolumeState.AVAILABLE, 'in-use': StorageVolumeState.INUSE, 'error': StorageVolumeState.ERROR, 'creating': StorageVolumeState.CREATING, 'deleting': StorageVolumeState.DELETING, 'deleted': StorageVolumeState.DELETED, 'error_deleting': StorageVolumeState.ERROR}
    SNAPSHOT_STATE_MAP = {'pending': VolumeSnapshotState.CREATING, 'completed': VolumeSnapshotState.AVAILABLE, 'error': VolumeSnapshotState.ERROR}

    def list_nodes(self, ex_node_ids=None, ex_filters=None):
        """
        Lists all nodes.

        Ex_node_ids parameter is used to filter the list of
        nodes that should be returned. Only the nodes
        with the corresponding node IDs will be returned.

        :param      ex_node_ids: List of ``node.id``
        :type       ex_node_ids: ``list`` of ``str``

        :param      ex_filters: The filters so that the list includes
                                information for certain nodes only.
        :type       ex_filters: ``dict``

        :rtype: ``list`` of :class:`Node`
        """
        params = {'Action': 'DescribeInstances'}
        if ex_node_ids:
            params.update(self._pathlist('InstanceId', ex_node_ids))
        if ex_filters:
            params.update(self._build_filters(ex_filters))
        elem = self.connection.request(self.path, params=params).object
        nodes = []
        for rs in findall(element=elem, xpath='reservationSet/item', namespace=NAMESPACE):
            nodes += self._to_nodes(rs, 'instancesSet/item')
        nodes_elastic_ips_mappings = self.ex_describe_addresses(nodes)
        for node in nodes:
            ips = nodes_elastic_ips_mappings[node.id]
            node.public_ips.extend(ips)
        return nodes

    def list_sizes(self, location=None):
        from libcloud.compute.constants.ec2_instance_types import INSTANCE_TYPES
        from libcloud.compute.constants.ec2_region_details_complete import REGION_DETAILS
        available_types = REGION_DETAILS[self.region_name]['instance_types']
        sizes = []
        for instance_type in available_types:
            attributes = INSTANCE_TYPES[instance_type]
            attributes = copy.deepcopy(attributes)
            try:
                price = get_size_price(driver_type='compute', driver_name='ec2_linux', size_id=instance_type, region=self.region_name)
                if price is None:
                    attributes['price'] = None
                else:
                    attributes['price'] = price
            except KeyError:
                attributes['price'] = None
            sizes.append(NodeSize(driver=self, **attributes))
        return sizes

    def list_images(self, location=None, ex_image_ids=None, ex_owner=None, ex_executableby=None, ex_filters=None):
        """
        Lists all images
        @inherits: :class:`NodeDriver.list_images`

        Ex_image_ids parameter is used to filter the list of
        images that should be returned. Only the images
        with the corresponding image IDs will be returned.

        Ex_owner parameter is used to filter the list of
        images that should be returned. Only the images
        with the corresponding owner will be returned.
        Valid values: amazon|aws-marketplace|self|all|aws id

        Ex_executableby parameter describes images for which
        the specified user has explicit launch permissions.
        The user can be an AWS account ID, self to return
        images for which the sender of the request has
        explicit launch permissions, or all to return
        images with public launch permissions.
        Valid values: all|self|aws id

        Ex_filters parameter is used to filter the list of
        images that should be returned. Only images matching
        the filter will be returned.

        :param      ex_image_ids: List of ``NodeImage.id``
        :type       ex_image_ids: ``list`` of ``str``

        :param      ex_owner: Owner name
        :type       ex_owner: ``str``

        :param      ex_executableby: Executable by
        :type       ex_executableby: ``str``

        :param      ex_filters: Filter by
        :type       ex_filters: ``dict``

        :rtype: ``list`` of :class:`NodeImage`
        """
        params = {'Action': 'DescribeImages'}
        if ex_owner:
            params.update({'Owner.1': ex_owner})
        if ex_executableby:
            params.update({'ExecutableBy.1': ex_executableby})
        if ex_image_ids:
            for index, image_id in enumerate(ex_image_ids):
                index += 1
                params.update({'ImageId.%s' % index: image_id})
        if ex_filters:
            params.update(self._build_filters(ex_filters))
        images = self._to_images(self.connection.request(self.path, params=params).object)
        return images

    def get_image(self, image_id):
        """
        Gets an image based on an image_id.

        :param image_id: Image identifier
        :type image_id: ``str``

        :return: A NodeImage object
        :rtype: :class:`NodeImage`

        """
        images = self.list_images(ex_image_ids=[image_id])
        image = images[0]
        return image

    def list_locations(self):
        locations = []
        iterator = enumerate(self.ex_list_availability_zones())
        for index, availability_zone in iterator:
            locations.append(EC2NodeLocation(index, availability_zone.name, self.country, self, availability_zone))
        return locations

    def list_volumes(self, node=None, ex_filters=None):
        """
        List volumes that are attached to a node, if specified and those that
        satisfy the filters, if specified.

        :param node: The node to which the volumes are attached.
        :type node: :class:`Node`

        :param ex_filters: The dictionary of additional filters.
        :type ex_filters: ``dict``

        :return: The list of volumes that match the criteria.
        :rtype: ``list`` of :class:`StorageVolume`
        """
        params = {'Action': 'DescribeVolumes'}
        if not ex_filters:
            ex_filters = {}
        if node:
            ex_filters['attachment.instance-id'] = node.id
        if node or ex_filters:
            params.update(self._build_filters(ex_filters))
        response = self.connection.request(self.path, params=params).object
        volumes = [self._to_volume(el) for el in response.findall(fixxpath(xpath='volumeSet/item', namespace=NAMESPACE))]
        return volumes

    def create_node(self, name, size, image, location=None, auth=None, ex_keyname=None, ex_userdata=None, ex_security_groups=None, ex_securitygroup=None, ex_security_group_ids=None, ex_metadata=None, ex_mincount=1, ex_maxcount=1, ex_clienttoken=None, ex_blockdevicemappings=None, ex_iamprofile=None, ex_ebs_optimized=None, ex_subnet=None, ex_placement_group=None, ex_assign_public_ip=False, ex_terminate_on_shutdown=False, ex_spot=False, ex_spot_max_price=None):
        """
        Create a new EC2 node.

        Reference: http://bit.ly/8ZyPSy [docs.amazonwebservices.com]

        @inherits: :class:`NodeDriver.create_node`

        :keyword    ex_keyname: The name of the key pair
        :type       ex_keyname: ``str``

        :keyword    ex_userdata: User data
        :type       ex_userdata: ``str``

        :keyword    ex_security_groups: A list of names of security groups to
                                        assign to the node.
        :type       ex_security_groups:   ``list``

        :keyword    ex_security_group_ids: A list of ids of security groups to
                                        assign to the node.[for VPC nodes only]
        :type       ex_security_group_ids:   ``list``

        :keyword    ex_metadata: Key/Value metadata to associate with a node
        :type       ex_metadata: ``dict``

        :keyword    ex_mincount: Minimum number of instances to launch
        :type       ex_mincount: ``int``

        :keyword    ex_maxcount: Maximum number of instances to launch
        :type       ex_maxcount: ``int``

        :keyword    ex_clienttoken: Unique identifier to ensure idempotency
        :type       ex_clienttoken: ``str``

        :keyword    ex_blockdevicemappings: ``list`` of ``dict`` block device
                    mappings.
        :type       ex_blockdevicemappings: ``list`` of ``dict``

        :keyword    ex_iam_profile: Name or ARN of IAM profile
        :type       ex_iam_profile: ``str``

        :keyword    ex_ebs_optimized: EBS-Optimized if True
        :type       ex_ebs_optimized: ``bool``

        :keyword    ex_subnet: The subnet to launch the instance into.
        :type       ex_subnet: :class:`.EC2Subnet`

        :keyword    ex_placement_group: The name of the placement group to
                                        launch the instance into.
        :type       ex_placement_group: ``str``

        :keyword    ex_assign_public_ip: If True, the instance will
                                         be assigned a public ip address.
                                         Note : It takes takes a short
                                         while for the instance to be
                                         assigned the public ip so the
                                         node returned will NOT have
                                         the public ip assigned yet.
        :type       ex_assign_public_ip: ``bool``

        :keyword    ex_terminate_on_shutdown: Indicates if the instance
                                              should be terminated instead
                                              of just shut down when using
                                              the operating systems command
                                              for system shutdown.
        :type       ex_terminate_on_shutdown: ``bool``

        :keyword    ex_spot: If true, ask for a Spot Instance instead of
                             requesting On-Demand.
        :type       ex_spot: ``bool``

        :keyword    ex_spot_max_price: Maximum price to pay for the spot
                                       instance. If not specified, the
                                       on-demand price will be used.
        :type       ex_spot_max_price: ``float``
        """
        params = {'Action': 'RunInstances', 'ImageId': image.id, 'MinCount': str(ex_mincount), 'MaxCount': str(ex_maxcount), 'InstanceType': size.id}
        if ex_terminate_on_shutdown:
            params['InstanceInitiatedShutdownBehavior'] = 'terminate'
        if ex_spot:
            params['InstanceMarketOptions.MarketType'] = 'spot'
            if ex_spot_max_price is not None:
                params['InstanceMarketOptions.SpotOptions.MaxPrice'] = str(ex_spot_max_price)
        if ex_security_groups and ex_securitygroup:
            raise ValueError('You can only supply ex_security_groups or ex_securitygroup')
        security_groups = ex_security_groups or ex_securitygroup
        if security_groups:
            if not isinstance(security_groups, (tuple, list)):
                security_groups = [security_groups]
            for sig in range(len(security_groups)):
                params['SecurityGroup.%d' % (sig + 1,)] = security_groups[sig]
        if ex_security_group_ids and (not ex_subnet):
            raise ValueError('You can only supply ex_security_group_ids combinated with ex_subnet')
        security_group_ids = ex_security_group_ids
        security_group_id_params = {}
        if security_group_ids:
            if not isinstance(security_group_ids, (tuple, list)):
                security_group_ids = [security_group_ids]
            for sig in range(len(security_group_ids)):
                security_group_id_params['SecurityGroupId.%d' % (sig + 1,)] = security_group_ids[sig]
        if location:
            availability_zone = getattr(location, 'availability_zone', None)
            if availability_zone:
                if availability_zone.region_name != self.region_name:
                    raise AttributeError('Invalid availability zone: %s' % availability_zone.name)
                params['Placement.AvailabilityZone'] = availability_zone.name
        if auth and ex_keyname:
            raise AttributeError('Cannot specify auth and ex_keyname together')
        if auth:
            auth = self._get_and_check_auth(auth)
            key = self.ex_find_or_import_keypair_by_key_material(auth.pubkey)
            params['KeyName'] = key['keyName']
        if ex_keyname:
            params['KeyName'] = ex_keyname
        if ex_userdata:
            params['UserData'] = base64.b64encode(b(ex_userdata)).decode('utf-8')
        if ex_clienttoken:
            params['ClientToken'] = ex_clienttoken
        if ex_blockdevicemappings:
            params.update(self._get_block_device_mapping_params(ex_blockdevicemappings))
        if ex_iamprofile:
            if not isinstance(ex_iamprofile, basestring):
                raise AttributeError('ex_iamprofile not string')
            if ex_iamprofile.startswith('arn:aws:iam:'):
                params['IamInstanceProfile.Arn'] = ex_iamprofile
            else:
                params['IamInstanceProfile.Name'] = ex_iamprofile
        if ex_ebs_optimized:
            params['EbsOptimized'] = ex_ebs_optimized
        subnet_id = None
        if ex_subnet:
            subnet_id = ex_subnet.id
        if ex_placement_group:
            params['Placement.GroupName'] = ex_placement_group
        assign_public_ip = ex_assign_public_ip
        if assign_public_ip:
            root_key = 'NetworkInterface.1.'
            params[root_key + 'AssociatePublicIpAddress'] = 'true'
            params[root_key + 'DeleteOnTermination'] = 'true'
            params[root_key + 'DeviceIndex'] = '0'
            if subnet_id:
                params[root_key + 'SubnetId'] = subnet_id
            for key, security_group_id in security_group_id_params.items():
                key = root_key + key
                params[key] = security_group_id
        else:
            params.update(security_group_id_params)
            if subnet_id:
                params['SubnetId'] = subnet_id
        tags = {'Name': name}
        if ex_metadata:
            tags.update(ex_metadata)
        tagspec_root = 'TagSpecification.1.'
        params[tagspec_root + 'ResourceType'] = 'instance'
        tag_nr = 1
        for k, v in tags.items():
            tag_root = tagspec_root + 'Tag.%d.' % tag_nr
            params[tag_root + 'Key'] = k
            params[tag_root + 'Value'] = v
            tag_nr += 1
        object = self.connection.request(self.path, params=params).object
        nodes = self._to_nodes(object, 'instancesSet/item')
        for node in nodes:
            node.name = name
            node.extra.update({'tags': tags})
        if len(nodes) == 1:
            return nodes[0]
        else:
            return nodes

    def reboot_node(self, node):
        params = {'Action': 'RebootInstances'}
        params.update(self._pathlist('InstanceId', [node.id]))
        res = self.connection.request(self.path, params=params).object
        return self._get_boolean(res)

    def destroy_node(self, node):
        params = {'Action': 'TerminateInstances'}
        params.update(self._pathlist('InstanceId', [node.id]))
        res = self.connection.request(self.path, params=params).object
        return self._get_terminate_boolean(res)

    def create_volume(self, size, name, location=None, snapshot=None, ex_volume_type='standard', ex_iops=None, ex_encrypted=False, ex_kms_key_id=None, ex_throughput=None):
        """
        Create a new volume.

        :param size: Size of volume in gigabytes (required)
        :type size: ``int``

        :param name: Name of the volume to be created
        :type name: ``str``

        :param location: Which data center to create a volume in. If
                               empty, undefined behavior will be selected.
                               (optional)
        :type location: :class:`.NodeLocation`

        :param snapshot:  Snapshot from which to create the new
                               volume.  (optional)
        :type snapshot:  :class:`.VolumeSnapshot`

        :param location: Datacenter in which to create a volume in.
        :type location: :class:`.ExEC2AvailabilityZone`

        :param ex_volume_type: Type of volume to create.
        :type ex_volume_type: ``str``

        :param ex_iops: The number of I/O operations per second (IOPS)
                     that the volume supports. Only used if ex_volume_type
                     is io1, io2 or gp3.
        :type ex_iops: ``int``

        :param ex_encrypted: Specifies whether the volume should be encrypted.
        :type ex_encrypted: ``bool``

        :param ex_kms_key_id: The full ARN of the AWS Key Management
                            Service (AWS KMS) customer master key (CMK) to use
                            when creating the encrypted volume.
                            Example:
                            arn:aws:kms:us-east-1:012345678910:key/abcd1234-a123
                            -456a-a12b-a123b4cd56ef.
                            Only used if encrypted is set to True.
        :type ex_kms_key_id: ``str``

        :param ex_throughput: The throughput to provision for a volume, with a
                            maximum of 1,000 MiB/s. Only used if ex_volume_type
                            is gp3.
        :type ex_throughput: ``int``

        :return: The newly created volume.
        :rtype: :class:`StorageVolume`
        """
        params = {'Action': 'CreateVolume', 'Size': str(size)}
        if ex_volume_type and ex_volume_type not in VALID_VOLUME_TYPES:
            raise ValueError('Invalid volume type specified: %s' % ex_volume_type)
        if snapshot:
            params['SnapshotId'] = snapshot.id
        if not location:
            location = self.list_locations()[0]
        params['AvailabilityZone'] = location.availability_zone.name
        if ex_volume_type:
            params['VolumeType'] = ex_volume_type
        if ex_volume_type in ['io1', 'io2', 'gp3'] and ex_iops:
            params['Iops'] = ex_iops
        if ex_encrypted:
            params['Encrypted'] = 1
            if ex_kms_key_id is not None:
                params['KmsKeyId'] = ex_kms_key_id
        if ex_volume_type == 'gp3' and ex_throughput:
            params['Throughput'] = ex_throughput
        volume = self._to_volume(self.connection.request(self.path, params=params).object, name=name)
        if self.ex_create_tags(volume, {'Name': name}):
            volume.extra['tags']['Name'] = name
        return volume

    def attach_volume(self, node, volume, device):
        params = {'Action': 'AttachVolume', 'VolumeId': volume.id, 'InstanceId': node.id, 'Device': device}
        self.connection.request(self.path, params=params)
        return True

    def detach_volume(self, volume, ex_force=False):
        params = {'Action': 'DetachVolume', 'VolumeId': volume.id}
        if ex_force:
            params['Force'] = 1
        self.connection.request(self.path, params=params)
        return True

    def destroy_volume(self, volume):
        params = {'Action': 'DeleteVolume', 'VolumeId': volume.id}
        response = self.connection.request(self.path, params=params).object
        return self._get_boolean(response)

    def create_volume_snapshot(self, volume, name=None, ex_metadata=None):
        """
        Create snapshot from volume

        :param      volume: Instance of ``StorageVolume``
        :type       volume: ``StorageVolume``

        :param      name: Name of snapshot (optional)
        :type       name: ``str``

        :keyword    ex_metadata: The Key/Value metadata to associate
                                 with a snapshot (optional)
        :type       ex_metadata: ``dict``

        :rtype: :class:`VolumeSnapshot`
        """
        params = {'Action': 'CreateSnapshot', 'VolumeId': volume.id}
        if name:
            params.update({'Description': name})
        if ex_metadata is None:
            ex_metadata = {}
        response = self.connection.request(self.path, params=params).object
        snapshot = self._to_snapshot(response, name)
        ex_metadata.update(**{'Name': name} if name else {})
        if self.ex_create_tags(snapshot, ex_metadata):
            snapshot.extra['tags'] = ex_metadata
        return snapshot

    def list_volume_snapshots(self, volume):
        return [snapshot for snapshot in self.list_snapshots(owner='self') if snapshot.extra['volume_id'] == volume.id]

    def list_snapshots(self, snapshot=None, owner=None):
        """
        Describes all snapshots.

        :param snapshot: If provided, only returns snapshot information for the
                         provided snapshot.

        :param owner: The owner of the snapshot: self|amazon|ID
        :type owner: ``str``

        :rtype: ``list`` of :class:`VolumeSnapshot`
        """
        params = {'Action': 'DescribeSnapshots'}
        if snapshot:
            params.update({'SnapshotId.1': snapshot.id})
        if owner:
            params.update({'Owner.1': owner})
        response = self.connection.request(self.path, params=params).object
        snapshots = self._to_snapshots(response)
        return snapshots

    def destroy_volume_snapshot(self, snapshot):
        params = {'Action': 'DeleteSnapshot', 'SnapshotId': snapshot.id}
        response = self.connection.request(self.path, params=params).object
        return self._get_boolean(response)

    def list_key_pairs(self):
        params = {'Action': 'DescribeKeyPairs'}
        response = self.connection.request(self.path, params=params)
        elems = findall(element=response.object, xpath='keySet/item', namespace=NAMESPACE)
        key_pairs = self._to_key_pairs(elems=elems)
        return key_pairs

    def get_key_pair(self, name):
        params = {'Action': 'DescribeKeyPairs', 'KeyName': name}
        response = self.connection.request(self.path, params=params)
        elems = findall(element=response.object, xpath='keySet/item', namespace=NAMESPACE)
        key_pair = self._to_key_pairs(elems=elems)[0]
        return key_pair

    def create_key_pair(self, name):
        params = {'Action': 'CreateKeyPair', 'KeyName': name}
        response = self.connection.request(self.path, params=params)
        elem = response.object
        key_pair = self._to_key_pair(elem=elem)
        return key_pair

    def import_key_pair_from_string(self, name, key_material):
        base64key = ensure_string(base64.b64encode(b(key_material)))
        params = {'Action': 'ImportKeyPair', 'KeyName': name, 'PublicKeyMaterial': base64key}
        response = self.connection.request(self.path, params=params)
        elem = response.object
        key_pair = self._to_key_pair(elem=elem)
        return key_pair

    def delete_key_pair(self, key_pair):
        params = {'Action': 'DeleteKeyPair', 'KeyName': key_pair.name}
        res = self.connection.request(self.path, params=params).object
        return self._get_boolean(res)

    def copy_image(self, image, source_region, name=None, description=None):
        """
        Copy an Amazon Machine Image from the specified source region
        to the current region.

        @inherits: :class:`NodeDriver.copy_image`

        :param      source_region: The region where the image resides
        :type       source_region: ``str``

        :param      image: Instance of class NodeImage
        :type       image: :class:`NodeImage`

        :param      name: The name of the new image
        :type       name: ``str``

        :param      description: The description of the new image
        :type       description: ``str``

        :return:    Instance of class ``NodeImage``
        :rtype:     :class:`NodeImage`
        """
        params = {'Action': 'CopyImage', 'SourceRegion': source_region, 'SourceImageId': image.id}
        if name is not None:
            params['Name'] = name
        if description is not None:
            params['Description'] = description
        image = self._to_image(self.connection.request(self.path, params=params).object)
        return image

    def create_image(self, node, name, description=None, reboot=False, block_device_mapping=None):
        """
        Create an Amazon Machine Image based off of an EBS-backed instance.

        @inherits: :class:`NodeDriver.create_image`

        :param      node: Instance of ``Node``
        :type       node: :class: `Node`

        :param      name: The name for the new image
        :type       name: ``str``

        :param      block_device_mapping: A dictionary of the disk layout
                                          An example of this dict is included
                                          below.
        :type       block_device_mapping: ``list`` of ``dict``

        :param      reboot: Whether or not to shutdown the instance before
                               creation. Amazon calls this NoReboot and
                               sets it to false by default to ensure a
                               clean image.
        :type       reboot: ``bool``

        :param      description: An optional description for the new image
        :type       description: ``str``

        An example block device mapping dictionary is included:

        mapping = [{'VirtualName': None,
                    'Ebs': {'VolumeSize': 10,
                            'VolumeType': 'standard',
                            'DeleteOnTermination': 'true'},
                            'DeviceName': '/dev/sda1'}]

        :return:    Instance of class ``NodeImage``
        :rtype:     :class:`NodeImage`
        """
        params = {'Action': 'CreateImage', 'InstanceId': node.id, 'Name': name, 'NoReboot': not reboot}
        if description is not None:
            params['Description'] = description
        if block_device_mapping is not None:
            params.update(self._get_block_device_mapping_params(block_device_mapping))
        image = self._to_image(self.connection.request(self.path, params=params).object)
        return image

    def delete_image(self, image):
        """
        Deletes an image at Amazon given a NodeImage object

        @inherits: :class:`NodeDriver.delete_image`

        :param image: Instance of ``NodeImage``
        :type image: :class: `NodeImage`

        :rtype:     ``bool``
        """
        params = {'Action': 'DeregisterImage', 'ImageId': image.id}
        response = self.connection.request(self.path, params=params).object
        return self._get_boolean(response)

    def start_node(self, node):
        """
        Starts the node by passing in the node object, does not work with
        instance store backed instances.

        :param      node: The node to be used
        :type       node: :class:`Node`

        :rtype: ``bool``
        """
        params = {'Action': 'StartInstances'}
        params.update(self._pathlist('InstanceId', [node.id]))
        res = self.connection.request(self.path, params=params).object
        return self._get_state_boolean(res)

    def stop_node(self, node):
        """
        Stops the node by passing in the node object, does not work with
        instance store backed instances

        :param      node: The node to be used
        :type       node: :class:`Node`

        :rtype: ``bool``
        """
        params = {'Action': 'StopInstances'}
        params.update(self._pathlist('InstanceId', [node.id]))
        res = self.connection.request(self.path, params=params).object
        return self._get_state_boolean(res)

    def ex_create_placement_group(self, name):
        """
        Creates a new placement group.

        :param name: The name for the new placement group
        :type name: ``str``

        :rtype: ``bool``
        """
        params = {'Action': 'CreatePlacementGroup', 'Strategy': 'cluster', 'GroupName': name}
        response = self.connection.request(self.path, params=params).object
        return self._get_boolean(response)

    def ex_delete_placement_group(self, name):
        """
        Deletes a placement group.

        :param name: The placement group name
        :type name: ``str``

        :rtype: ``bool``
        """
        params = {'Action': 'DeletePlacementGroup', 'GroupName': name}
        response = self.connection.request(self.path, params=params).object
        return self._get_boolean(response)

    def ex_import_snapshot(self, client_data=None, client_token=None, description=None, disk_container=None, dry_run=None, role_name=None):
        """
        Imports a disk into an EBS snapshot. More information can be found
        at https://goo.gl/sbXkYA.

        :param  client_data: Describes the client specific data (optional)
        :type   client_data: ``dict``

        :param  client_token: The token to enable idempotency for VM
                              (optional)
        :type   client_token: ``str``

        :param  description: The description string for the
                             import snapshot task.(optional)
        :type   description: ``str``

        :param  disk_container:The disk container object for the
                              import snapshot request.
        :type   disk_container:``dict``

        :param  dry_run: Checks whether you have the permission for
                        the action, without actually making the request,
                        and provides an error response.(optional)
        :type   dry_run: ``bool``

        :param  role_name: The name of the role to use when not using the
                          default role, 'vmimport'.(optional)
        :type   role_name: ``str``

        :rtype: :class: ``VolumeSnapshot``
        """
        params = {'Action': 'ImportSnapshot'}
        if client_token is not None:
            params['ClientToken'] = client_token
        if description is not None:
            params['Description'] = description
        if disk_container is not None:
            params.update(self._get_disk_container_params(disk_container))
        if dry_run is not None:
            params['DryRun'] = dry_run
        if role_name is not None:
            params['RoleName'] = role_name
        importSnapshot = self.connection.request(self.path, params=params).object
        importTaskId = findtext(element=importSnapshot, xpath='importTaskId', namespace=NAMESPACE)
        volumeSnapshot = self._wait_for_import_snapshot_completion(import_task_id=importTaskId, timeout=1800, interval=15)
        return volumeSnapshot

    def _wait_for_import_snapshot_completion(self, import_task_id, timeout=1800, interval=15):
        """
        It waits for import snapshot to be completed

        :param import_task_id: Import task Id for the
                               current Import Snapshot Task
        :type import_task_id: ``str``

        :param timeout: Timeout value for snapshot generation
        :type timeout: ``float``

        :param interval: Time interval for repetitive describe
                         import snapshot tasks requests
        :type interval: ``float``

        :rtype: :class:``VolumeSnapshot``
        """
        start_time = time.time()
        snapshotId = None
        while snapshotId is None:
            if time.time() - start_time >= timeout:
                raise Exception('Timeout while waiting for import task Id %s' % import_task_id)
            res = self.ex_describe_import_snapshot_tasks(import_task_id)
            snapshotId = res.snapshotId
            if snapshotId is None:
                time.sleep(interval)
        volumeSnapshot = VolumeSnapshot(snapshotId, driver=self)
        return volumeSnapshot

    def ex_describe_import_snapshot_tasks(self, import_task_id, dry_run=None):
        """
        Describes your import snapshot tasks. More information can be found
        at https://goo.gl/CI0MdS.

        :param import_task_id: Import task Id for the current
                               Import Snapshot Task
        :type import_task_id: ``str``

        :param  dry_run: Checks whether you have the permission for
                        the action, without actually making the request,
                        and provides an error response.(optional)
        :type   dry_run: ``bool``

        :rtype: :class:``DescribeImportSnapshotTasks Object``

        """
        params = {'Action': 'DescribeImportSnapshotTasks'}
        if dry_run is not None:
            params['DryRun'] = dry_run
        params['ImportTaskId.1'] = import_task_id
        res = self._to_import_snapshot_task(self.connection.request(self.path, params=params).object)
        return res

    def ex_list_placement_groups(self, names=None):
        """
        A list of placement groups.

        :param names: Placement Group names
        :type names: ``list`` of ``str``

        :rtype: ``list`` of :class:`.EC2PlacementGroup`
        """
        names = names or []
        params = {'Action': 'DescribePlacementGroups'}
        for index, name in enumerate(names):
            params['GroupName.%s' % index + 1] = name
        response = self.connection.request(self.path, params=params).object
        return self._to_placement_groups(response)

    def ex_register_image(self, name, description=None, architecture=None, image_location=None, root_device_name=None, block_device_mapping=None, kernel_id=None, ramdisk_id=None, virtualization_type=None, ena_support=None, billing_products=None, sriov_net_support=None, boot_mode: str=None, tpm_support: str=None, uefi_data: str=None, imds_support: str=None):
        """
        Registers an Amazon Machine Image based off of an EBS-backed instance.
        Can also be used to create images from snapshots. More information
        can be found at http://goo.gl/hqZq0a.

        :param      name:  The name for the AMI being registered
        :type       name: ``str``

        :param      description: The description of the AMI (optional)
        :type       description: ``str``

        :param      architecture: The architecture of the AMI (i386/x86_64)
                                  (optional)
        :type       architecture: ``str``

        :param      image_location: The location of the AMI within Amazon S3
                                    Required if registering an instance
                                    store-backed AMI
        :type       image_location: ``str``

        :param      root_device_name: The device name for the root device
                                      Required if registering an EBS-backed AMI
        :type       root_device_name: ``str``

        :param      block_device_mapping: A dictionary of the disk layout
                                          (optional)
        :type       block_device_mapping: ``dict``

        :param      kernel_id: Kernel id for AMI (optional)
        :type       kernel_id: ``str``

        :param      ramdisk_id: RAM disk for AMI (optional)
        :type       ramdisk_id: ``str``

        :param      virtualization_type: The type of virtualization for the
                                         AMI you are registering, paravirt
                                         or hvm (optional)
        :type       virtualization_type: ``str``

        :param      ena_support: Enable enhanced networking with Elastic
                                 Network Adapter for the AMI
        :type       ena_support: ``bool``

        :param      billing_products: The billing product codes
        :type       billing_products: ''list''

        :param      sriov_net_support: Set to "simple" to enable enhanced
                                       networking with the Intel 82599 Virtual
                                       Function interface
        :type       sriov_net_support: ``str``

        :param      boot_mode: desired boot mode for the AMI.
        :type       boot_mode: ``str``

        :param      tpm_support: set to ``v2.0`` to enable TPM support.
        :type       tpm_support: ``str``

        :param      uefi_data: base64 representation of the non-volatile UEFI
                               variable store.
        :type       uefi_data: ``str``

        :param      imds_support: set to ``v2.0`` to require HTTP tokens when
                                  accessing the IMDS on instances launched
                                  from the resulting image.
        :type       imds_support: ``str``

        :rtype:     :class:`NodeImage`
        """
        params = {'Action': 'RegisterImage', 'Name': name}
        if description is not None:
            params['Description'] = description
        if architecture is not None:
            params['Architecture'] = architecture
        if image_location is not None:
            params['ImageLocation'] = image_location
        if root_device_name is not None:
            params['RootDeviceName'] = root_device_name
        if block_device_mapping is not None:
            params.update(self._get_block_device_mapping_params(block_device_mapping))
        if kernel_id is not None:
            params['KernelId'] = kernel_id
        if ramdisk_id is not None:
            params['RamDiskId'] = ramdisk_id
        if virtualization_type is not None:
            params['VirtualizationType'] = virtualization_type
        if ena_support is not None:
            params['EnaSupport'] = ena_support
        if billing_products is not None:
            params.update(self._get_billing_product_params(billing_products))
        if sriov_net_support is not None:
            params['SriovNetSupport'] = sriov_net_support
        if boot_mode is not None:
            params['BootMode'] = boot_mode
        if tpm_support is not None:
            params['TpmSupport'] = tpm_support
        if uefi_data is not None:
            params['UefiData'] = uefi_data
        if imds_support is not None:
            params['ImdsSupport'] = imds_support
        image = self._to_image(self.connection.request(self.path, params=params).object)
        return image

    def ex_list_networks(self, network_ids=None, filters=None):
        """
        Returns a list of :class:`EC2Network` objects for the
        current region.

        :param      network_ids: Returns only networks matching the provided
                                 network IDs. If not specified, a list of all
                                 the networks in the corresponding region
                                 is returned.
        :type       network_ids: ``list``

        :param      filters: The filters so that the list returned includes
                             information for certain networks only.
        :type       filters: ``dict``

        :rtype:     ``list`` of :class:`EC2Network`
        """
        params = {'Action': 'DescribeVpcs'}
        if network_ids:
            params.update(self._pathlist('VpcId', network_ids))
        if filters:
            params.update(self._build_filters(filters))
        return self._to_networks(self.connection.request(self.path, params=params).object)

    def ex_create_network(self, cidr_block, name=None, instance_tenancy='default'):
        """
        Create a network/VPC

        :param      cidr_block: The CIDR block assigned to the network
        :type       cidr_block: ``str``

        :param      name: An optional name for the network
        :type       name: ``str``

        :param      instance_tenancy: The allowed tenancy of instances launched
                                      into the VPC.
                                      Valid values: default/dedicated
        :type       instance_tenancy: ``str``

        :return:    Dictionary of network properties
        :rtype:     ``dict``
        """
        params = {'Action': 'CreateVpc', 'CidrBlock': cidr_block, 'InstanceTenancy': instance_tenancy}
        response = self.connection.request(self.path, params=params).object
        element = response.findall(fixxpath(xpath='vpc', namespace=NAMESPACE))[0]
        network = self._to_network(element, name)
        if name and self.ex_create_tags(network, {'Name': name}):
            network.extra['tags']['Name'] = name
        return network

    def ex_delete_network(self, vpc):
        """
        Deletes a network/VPC.

        :param      vpc: VPC to delete.
        :type       vpc: :class:`.EC2Network`

        :rtype:     ``bool``
        """
        params = {'Action': 'DeleteVpc', 'VpcId': vpc.id}
        res = self.connection.request(self.path, params=params).object
        return self._get_boolean(res)

    def ex_list_subnets(self, subnet_ids=None, filters=None):
        """
        Returns a list of :class:`EC2NetworkSubnet` objects for the
        current region.

        :param      subnet_ids: Returns only subnets matching the provided
                                subnet IDs. If not specified, a list of all
                                the subnets in the corresponding region
                                is returned.
        :type       subnet_ids: ``list``

        :param      filters: The filters so that the list returned includes
                             information for certain subnets only.
        :type       filters: ``dict``

        :rtype:     ``list`` of :class:`EC2NetworkSubnet`
        """
        params = {'Action': 'DescribeSubnets'}
        if subnet_ids:
            params.update(self._pathlist('SubnetId', subnet_ids))
        if filters:
            params.update(self._build_filters(filters))
        return self._to_subnets(self.connection.request(self.path, params=params).object)

    def ex_create_subnet(self, vpc_id, cidr_block, availability_zone, name=None):
        """
        Creates a network subnet within a VPC.

        :param      vpc_id: The ID of the VPC that the subnet should be
                            associated with
        :type       vpc_id: ``str``

        :param      cidr_block: The CIDR block assigned to the subnet
        :type       cidr_block: ``str``

        :param      availability_zone: The availability zone where the subnet
                                       should reside
        :type       availability_zone: ``str``

        :param      name: An optional name for the network
        :type       name: ``str``

        :rtype:     :class: `EC2NetworkSubnet`
        """
        params = {'Action': 'CreateSubnet', 'VpcId': vpc_id, 'CidrBlock': cidr_block, 'AvailabilityZone': availability_zone}
        response = self.connection.request(self.path, params=params).object
        element = response.findall(fixxpath(xpath='subnet', namespace=NAMESPACE))[0]
        subnet = self._to_subnet(element, name)
        if name and self.ex_create_tags(subnet, {'Name': name}):
            subnet.extra['tags']['Name'] = name
        return subnet

    def ex_modify_subnet_attribute(self, subnet, attribute='auto_public_ip', value=False):
        """
        Modifies a subnet attribute.
        You can only modify one attribute at a time.

        :param      subnet: The subnet to delete
        :type       subnet: :class:`.EC2NetworkSubnet`

        :param      attribute: The attribute to set on the subnet; one of:
                               ``'auto_public_ip'``: Automatically allocate a
                               public IP address when a server is created
                               ``'auto_ipv6'``: Automatically assign an IPv6
                               address when a server is created
        :type       attribute: ``str``

        :param      value: The value to set the subnet attribute to
                           (defaults to ``False``)
        :type       value: ``bool``

        :rtype:     ``bool``
        """
        params = {'Action': 'ModifySubnetAttribute', 'SubnetId': subnet.id}
        if attribute == 'auto_public_ip':
            params['MapPublicIpOnLaunch.Value'] = value
        elif attribute == 'auto_ipv6':
            params['AssignIpv6AddressOnCreation.Value'] = value
        else:
            raise ValueError('Unsupported attribute: %s' % attribute)
        res = self.connection.request(self.path, params=params).object
        return self._get_boolean(res)

    def ex_delete_subnet(self, subnet):
        """
        Deletes a VPC subnet.

        :param      subnet: The subnet to delete
        :type       subnet: :class:`.EC2NetworkSubnet`

        :rtype:     ``bool``
        """
        params = {'Action': 'DeleteSubnet', 'SubnetId': subnet.id}
        res = self.connection.request(self.path, params=params).object
        return self._get_boolean(res)

    def ex_list_security_groups(self):
        """
        Lists existing Security Groups.

        @note: This is a non-standard extension API, and only works for EC2.

        :rtype: ``list`` of ``str``
        """
        params = {'Action': 'DescribeSecurityGroups'}
        response = self.connection.request(self.path, params=params).object
        groups = []
        for group in findall(element=response, xpath='securityGroupInfo/item', namespace=NAMESPACE):
            name = findtext(element=group, xpath='groupName', namespace=NAMESPACE)
            groups.append(name)
        return groups

    def ex_get_security_groups(self, group_ids=None, group_names=None, filters=None):
        """
        Returns a list of :class:`EC2SecurityGroup` objects for the
        current region.

        :param      group_ids: Returns only groups matching the provided
                               group IDs.
        :type       group_ids: ``list``

        :param      group_names: Returns only groups matching the provided
                                 group names.
        :type       group_ids: ``list``

        :param      filters: The filters so that the list returned includes
                             information for specific security groups only.
        :type       filters: ``dict``

        :rtype:     ``list`` of :class:`EC2SecurityGroup`
        """
        params = {'Action': 'DescribeSecurityGroups'}
        if group_ids:
            params.update(self._pathlist('GroupId', group_ids))
        if group_names:
            for name_idx, group_name in enumerate(group_names):
                name_idx += 1
                name_key = 'GroupName.%s' % name_idx
                params[name_key] = group_name
        if filters:
            params.update(self._build_filters(filters))
        response = self.connection.request(self.path, params=params)
        return self._to_security_groups(response.object)

    def ex_create_security_group(self, name, description, vpc_id=None):
        """
        Creates a new Security Group in EC2-Classic or a targeted VPC.

        :param      name:        The name of the security group to create.
                                 This must be unique.
        :type       name:        ``str``

        :param      description: Human readable description of a Security
                                 Group.
        :type       description: ``str``

        :param      vpc_id:      Optional identifier for VPC networks
        :type       vpc_id:      ``str``

        :rtype: ``dict``
        """
        params = {'Action': 'CreateSecurityGroup', 'GroupName': name, 'GroupDescription': description}
        if vpc_id is not None:
            params['VpcId'] = vpc_id
        response = self.connection.request(self.path, params=params).object
        group_id = findattr(element=response, xpath='groupId', namespace=NAMESPACE)
        return {'group_id': group_id}

    def ex_delete_security_group_by_id(self, group_id):
        """
        Deletes a new Security Group using the group ID.

        :param      group_id: The ID of the security group
        :type       group_id: ``str``

        :rtype: ``bool``
        """
        params = {'Action': 'DeleteSecurityGroup', 'GroupId': group_id}
        res = self.connection.request(self.path, params=params).object
        return self._get_boolean(res)

    def ex_delete_security_group_by_name(self, group_name):
        """
        Deletes a new Security Group using the group name.

        :param      group_name: The name of the security group
        :type       group_name: ``str``

        :rtype: ``bool``
        """
        params = {'Action': 'DeleteSecurityGroup', 'GroupName': group_name}
        res = self.connection.request(self.path, params=params).object
        return self._get_boolean(res)

    def ex_delete_security_group(self, name):
        """
        A wrapper method which calls ex_delete_security_group_by_name.

        :param      name: The name of the security group
        :type       name: ``str``

        :rtype: ``bool``
        """
        return self.ex_delete_security_group_by_name(name)

    def ex_authorize_security_group(self, name, from_port, to_port, cidr_ip, protocol='tcp'):
        """
        Edit a Security Group to allow specific traffic.

        @note: This is a non-standard extension API, and only works for EC2.

        :param      name: The name of the security group to edit
        :type       name: ``str``

        :param      from_port: The beginning of the port range to open
        :type       from_port: ``str``

        :param      to_port: The end of the port range to open
        :type       to_port: ``str``

        :param      cidr_ip: The ip to allow traffic for.
        :type       cidr_ip: ``str``

        :param      protocol: tcp/udp/icmp
        :type       protocol: ``str``

        :rtype: ``bool``
        """
        params = {'Action': 'AuthorizeSecurityGroupIngress', 'GroupName': name, 'IpProtocol': protocol, 'FromPort': str(from_port), 'ToPort': str(to_port), 'CidrIp': cidr_ip}
        try:
            res = self.connection.request(self.path, params=params.copy()).object
            return self._get_boolean(res)
        except Exception as e:
            if e.args[0].find('InvalidPermission.Duplicate') == -1:
                raise e

    def ex_authorize_security_group_ingress(self, id, from_port, to_port, cidr_ips=None, group_pairs=None, protocol='tcp', description=None):
        """
        Edit a Security Group to allow specific ingress traffic using
        CIDR blocks or either a group ID, group name or user ID (account).

        :param      id: The id of the security group to edit
        :type       id: ``str``

        :param      from_port: The beginning of the port range to open
        :type       from_port: ``int``

        :param      to_port: The end of the port range to open
        :type       to_port: ``int``

        :param      cidr_ips: The list of IP ranges to allow traffic for.
        :type       cidr_ips: ``list``

        :param      group_pairs: Source user/group pairs to allow traffic for.
                    More info can be found at http://goo.gl/stBHJF

                    EC2 Classic Example: To allow access from any system
                    associated with the default group on account 1234567890

                    [{'group_name': 'default', 'user_id': '1234567890'}]

                    VPC example: To allow access from any system associated
                    with security group sg-47ad482e on your own account

                    [{'group_id': ' sg-47ad482e'}]
        :type       group_pairs: ``list`` of ``dict``

        :param      protocol: tcp/udp/icmp
        :type       protocol: ``str``

        :param      description: description to be added to the rules inserted
        :type       description: ``str``

        :rtype: ``bool``
        """
        params = self._get_common_security_group_params(id, protocol, from_port, to_port, cidr_ips, group_pairs, description)
        params['Action'] = 'AuthorizeSecurityGroupIngress'
        res = self.connection.request(self.path, params=params).object
        return self._get_boolean(res)

    def ex_authorize_security_group_egress(self, id, from_port, to_port, cidr_ips, group_pairs=None, protocol='tcp'):
        """
        Edit a Security Group to allow specific egress traffic using
        CIDR blocks or either a group ID, group name or user ID (account).
        This call is not supported for EC2 classic and only works for VPC
        groups.

        :param      id: The id of the security group to edit
        :type       id: ``str``

        :param      from_port: The beginning of the port range to open
        :type       from_port: ``int``

        :param      to_port: The end of the port range to open
        :type       to_port: ``int``

        :param      cidr_ips: The list of ip ranges to allow traffic for.
        :type       cidr_ips: ``list``

        :param      group_pairs: Source user/group pairs to allow traffic for.
                    More info can be found at http://goo.gl/stBHJF

                    EC2 Classic Example: To allow access from any system
                    associated with the default group on account 1234567890

                    [{'group_name': 'default', 'user_id': '1234567890'}]

                    VPC Example: Allow access from any system associated with
                    security group sg-47ad482e on your own account

                    [{'group_id': ' sg-47ad482e'}]
        :type       group_pairs: ``list`` of ``dict``

        :param      protocol: tcp/udp/icmp
        :type       protocol: ``str``

        :rtype: ``bool``
        """
        params = self._get_common_security_group_params(id, protocol, from_port, to_port, cidr_ips, group_pairs)
        params['Action'] = 'AuthorizeSecurityGroupEgress'
        res = self.connection.request(self.path, params=params).object
        return self._get_boolean(res)

    def ex_revoke_security_group_ingress(self, id, from_port, to_port, cidr_ips=None, group_pairs=None, protocol='tcp'):
        """
        Edits a Security Group to revoke specific ingress traffic using
        CIDR blocks or either a group ID, group name or user ID (account).

        :param      id: The ID of the security group to edit
        :type       id: ``str``

        :param      from_port: The beginning of the port range to open
        :type       from_port: ``int``

        :param      to_port: The end of the port range to open
        :type       to_port: ``int``

        :param      cidr_ips: The list of ip ranges to allow traffic for.
        :type       cidr_ips: ``list``

        :param      group_pairs: Source user/group pairs to allow traffic for.
                    More info can be found at http://goo.gl/stBHJF

                    EC2 Classic Example: To allow access from any system
                    associated with the default group on account 1234567890

                    [{'group_name': 'default', 'user_id': '1234567890'}]

                    VPC Example: Allow access from any system associated with
                    security group sg-47ad482e on your own account

                    [{'group_id': ' sg-47ad482e'}]
        :type       group_pairs: ``list`` of ``dict``

        :param      protocol: tcp/udp/icmp
        :type       protocol: ``str``

        :rtype: ``bool``
        """
        params = self._get_common_security_group_params(id, protocol, from_port, to_port, cidr_ips, group_pairs)
        params['Action'] = 'RevokeSecurityGroupIngress'
        res = self.connection.request(self.path, params=params).object
        return self._get_boolean(res)

    def ex_revoke_security_group_egress(self, id, from_port, to_port, cidr_ips=None, group_pairs=None, protocol='tcp'):
        """
        Edit a Security Group to revoke specific egress traffic using
        CIDR blocks or either a group ID, group name or user ID (account).
        This call is not supported for EC2 classic and only works for
        VPC groups.

        :param      id: The id of the security group to edit
        :type       id: ``str``

        :param      from_port: The beginning of the port range to open
        :type       from_port: ``int``

        :param      to_port: The end of the port range to open
        :type       to_port: ``int``

        :param      cidr_ips: The list of ip ranges to allow traffic for.
        :type       cidr_ips: ``list``

        :param      group_pairs: Source user/group pairs to allow traffic for.
                    More info can be found at http://goo.gl/stBHJF

                    EC2 Classic Example: To allow access from any system
                    associated with the default group on account 1234567890

                    [{'group_name': 'default', 'user_id': '1234567890'}]

                    VPC Example: Allow access from any system associated with
                    security group sg-47ad482e on your own account

                    [{'group_id': ' sg-47ad482e'}]
        :type       group_pairs: ``list`` of ``dict``

        :param      protocol: tcp/udp/icmp
        :type       protocol: ``str``

        :rtype: ``bool``
        """
        params = self._get_common_security_group_params(id, protocol, from_port, to_port, cidr_ips, group_pairs)
        params['Action'] = 'RevokeSecurityGroupEgress'
        res = self.connection.request(self.path, params=params).object
        return self._get_boolean(res)

    def ex_authorize_security_group_permissive(self, name):
        """
        Edit a Security Group to allow all traffic.

        @note: This is a non-standard extension API, and only works for EC2.

        :param      name: The name of the security group to edit
        :type       name: ``str``

        :rtype: ``list`` of ``str``
        """
        results = []
        params = {'Action': 'AuthorizeSecurityGroupIngress', 'GroupName': name, 'IpProtocol': 'tcp', 'FromPort': '0', 'ToPort': '65535', 'CidrIp': '0.0.0.0/0'}
        try:
            results.append(self.connection.request(self.path, params=params.copy()).object)
        except Exception as e:
            if e.args[0].find('InvalidPermission.Duplicate') == -1:
                raise e
        params['IpProtocol'] = 'udp'
        try:
            results.append(self.connection.request(self.path, params=params.copy()).object)
        except Exception as e:
            if e.args[0].find('InvalidPermission.Duplicate') == -1:
                raise e
        params.update({'IpProtocol': 'icmp', 'FromPort': '-1', 'ToPort': '-1'})
        try:
            results.append(self.connection.request(self.path, params=params.copy()).object)
        except Exception as e:
            if e.args[0].find('InvalidPermission.Duplicate') == -1:
                raise e
        return results

    def ex_list_availability_zones(self, only_available=True):
        """
        Returns a list of :class:`ExEC2AvailabilityZone` objects for the
        current region.

        Note: This is an extension method and is only available for EC2
        driver.

        :keyword  only_available: If true, returns only availability zones
                                  with state 'available'
        :type     only_available: ``str``

        :rtype: ``list`` of :class:`ExEC2AvailabilityZone`
        """
        params = {'Action': 'DescribeAvailabilityZones'}
        filters = {'region-name': self.region_name}
        if only_available:
            filters['state'] = 'available'
        params.update(self._build_filters(filters))
        result = self.connection.request(self.path, params=params.copy()).object
        availability_zones = []
        for element in findall(element=result, xpath='availabilityZoneInfo/item', namespace=NAMESPACE):
            name = findtext(element=element, xpath='zoneName', namespace=NAMESPACE)
            zone_state = findtext(element=element, xpath='zoneState', namespace=NAMESPACE)
            region_name = findtext(element=element, xpath='regionName', namespace=NAMESPACE)
            availability_zone = ExEC2AvailabilityZone(name=name, zone_state=zone_state, region_name=region_name)
            availability_zones.append(availability_zone)
        return availability_zones

    def ex_describe_tags(self, resource):
        """
        Returns a dictionary of tags for a resource (e.g. Node or
        StorageVolume).

        :param  resource: The resource to be used
        :type   resource: any resource class, such as :class:`Node,`
                :class:`StorageVolume,` or :class:NodeImage`

        :return: A dictionary of Node tags
        :rtype: ``dict``
        """
        params = {'Action': 'DescribeTags'}
        filters = {'resource-id': resource.id}
        params.update(self._build_filters(filters))
        result = self.connection.request(self.path, params=params).object
        return self._get_resource_tags(result)

    def ex_create_tags(self, resource, tags):
        """
        Creates tags for a resource (Node or StorageVolume).

        :param resource: The resource to be tagged
        :type resource: :class:`Node` or :class:`StorageVolume` or
                        :class:`VolumeSnapshot`

        :param tags: A dictionary or other mapping of strings to strings,
                     associating tag names with tag values.
        :type tags: ``dict``

        :rtype: ``bool``
        """
        if not tags:
            return
        params = {'Action': 'CreateTags', 'ResourceId.0': resource.id}
        for i, key in enumerate(tags):
            params['Tag.%d.Key' % i] = key
            params['Tag.%d.Value' % i] = tags[key]
        res = self.connection.request(self.path, params=params.copy()).object
        return self._get_boolean(res)

    def ex_delete_tags(self, resource, tags):
        """
        Deletes tags from a resource.

        :param resource: The resource to be tagged
        :type resource: :class:`Node` or :class:`StorageVolume`

        :param tags: A dictionary or other mapping of strings to strings,
                     specifying the tag names and tag values to be deleted.
        :type tags: ``dict``

        :rtype: ``bool``
        """
        if not tags:
            return
        params = {'Action': 'DeleteTags', 'ResourceId.0': resource.id}
        for i, key in enumerate(tags):
            params['Tag.%d.Key' % i] = key
            if tags[key] is not None:
                params['Tag.%d.Value' % i] = tags[key]
        res = self.connection.request(self.path, params=params.copy()).object
        return self._get_boolean(res)

    def ex_get_metadata_for_node(self, node):
        """
        Returns the metadata associated with the node.

        :param      node: Node instance
        :type       node: :class:`Node`

        :return: A dictionary or other mapping of strings to strings,
                 associating tag names with tag values.
        :rtype tags: ``dict``
        """
        return node.extra['tags']

    def ex_allocate_address(self, domain='standard'):
        """
        Allocate a new Elastic IP address for EC2 classic or VPC

        :param      domain: The domain to allocate the new address in
                            (standard/vpc)
        :type       domain: ``str``

        :return:    Instance of ElasticIP
        :rtype:     :class:`ElasticIP`
        """
        params = {'Action': 'AllocateAddress'}
        if domain == 'vpc':
            params['Domain'] = domain
        response = self.connection.request(self.path, params=params).object
        return self._to_address(response, only_associated=False)

    def ex_release_address(self, elastic_ip, domain=None):
        """
        Releases an Elastic IP address using the IP (EC2-Classic) or
        using the allocation ID (VPC).

        :param      elastic_ip: Elastic IP instance
        :type       elastic_ip: :class:`ElasticIP`

        :param      domain: The domain where the IP resides (vpc only)
        :type       domain: ``str``

        :return:    True on success, False otherwise.
        :rtype:     ``bool``
        """
        params = {'Action': 'ReleaseAddress'}
        if domain is not None and domain != 'vpc':
            raise AttributeError('Domain can only be set to vpc')
        if domain is None:
            params['PublicIp'] = elastic_ip.ip
        else:
            params['AllocationId'] = elastic_ip.extra['allocation_id']
        response = self.connection.request(self.path, params=params).object
        return self._get_boolean(response)

    def ex_describe_all_addresses(self, only_associated=False):
        """
        Returns all the Elastic IP addresses for this account
        optionally, returns only addresses associated with nodes.

        :param    only_associated: If true, return only the addresses
                                   that are associated with an instance.
        :type     only_associated: ``bool``

        :return:  List of Elastic IP addresses.
        :rtype:   ``list`` of :class:`ElasticIP`
        """
        params = {'Action': 'DescribeAddresses'}
        response = self.connection.request(self.path, params=params).object
        return self._to_addresses(response, only_associated)

    def ex_associate_address_with_node(self, node, elastic_ip, domain=None):
        """
        Associate an Elastic IP address with a particular node.

        :param      node: Node instance
        :type       node: :class:`Node`

        :param      elastic_ip: Elastic IP instance
        :type       elastic_ip: :class:`ElasticIP`

        :param      domain: The domain where the IP resides (vpc only)
        :type       domain: ``str``

        :return:    A string representation of the association ID which is
                    required for VPC disassociation. EC2/standard
                    addresses return None
        :rtype:     ``None`` or ``str``
        """
        params = {'Action': 'AssociateAddress', 'InstanceId': node.id}
        if domain is not None and domain != 'vpc':
            raise AttributeError('Domain can only be set to vpc')
        if domain is None:
            params.update({'PublicIp': elastic_ip.ip})
        else:
            params.update({'AllocationId': elastic_ip.extra['allocation_id']})
        response = self.connection.request(self.path, params=params).object
        association_id = findtext(element=response, xpath='associationId', namespace=NAMESPACE)
        return association_id

    def ex_associate_addresses(self, node, elastic_ip, domain=None):
        """
        Note: This method has been deprecated in favor of
        the ex_associate_address_with_node method.
        """
        return self.ex_associate_address_with_node(node=node, elastic_ip=elastic_ip, domain=domain)

    def ex_disassociate_address(self, elastic_ip, domain=None):
        """
        Disassociates an Elastic IP address using the IP (EC2-Classic)
        or the association ID (VPC).

        :param      elastic_ip: ElasticIP instance
        :type       elastic_ip: :class:`ElasticIP`

        :param      domain: The domain where the IP resides (vpc only)
        :type       domain: ``str``

        :return:    True on success, False otherwise.
        :rtype:     ``bool``
        """
        params = {'Action': 'DisassociateAddress'}
        if domain is not None and domain != 'vpc':
            raise AttributeError('Domain can only be set to vpc')
        if domain is None:
            params['PublicIp'] = elastic_ip.ip
        else:
            params['AssociationId'] = elastic_ip.extra['association_id']
        res = self.connection.request(self.path, params=params).object
        return self._get_boolean(res)

    def ex_describe_addresses(self, nodes):
        """
        Returns Elastic IP addresses for all the nodes in the provided list.

        :param      nodes: A list of :class:`Node` instances
        :type       nodes: ``list`` of :class:`Node`

        :return:    Dictionary where a key is a node ID and the value is a
                    list with the Elastic IP addresses associated with
                    this node.
        :rtype:     ``dict``
        """
        if not nodes:
            return {}
        params = {'Action': 'DescribeAddresses'}
        if len(nodes) == 1:
            self._add_instance_filter(params, nodes[0])
        result = self.connection.request(self.path, params=params).object
        node_instance_ids = [node.id for node in nodes]
        nodes_elastic_ip_mappings = {}
        only_associated = True
        for node_id in node_instance_ids:
            nodes_elastic_ip_mappings.setdefault(node_id, [])
            for addr in self._to_addresses(result, only_associated):
                instance_id = addr.instance_id
                if node_id == instance_id:
                    nodes_elastic_ip_mappings[instance_id].append(addr.ip)
        return nodes_elastic_ip_mappings

    def ex_describe_addresses_for_node(self, node):
        """
        Returns a list of Elastic IP Addresses associated with this node.

        :param      node: Node instance
        :type       node: :class:`Node`

        :return: List Elastic IP Addresses attached to this node.
        :rtype: ``list`` of ``str``
        """
        node_elastic_ips = self.ex_describe_addresses([node])
        return node_elastic_ips[node.id]

    def ex_list_network_interfaces(self):
        """
        Returns all network interfaces.

        :return:    List of EC2NetworkInterface instances
        :rtype:     ``list`` of :class `EC2NetworkInterface`
        """
        params = {'Action': 'DescribeNetworkInterfaces'}
        return self._to_interfaces(self.connection.request(self.path, params=params).object)

    def ex_create_network_interface(self, subnet, name=None, description=None, private_ip_address=None):
        """
        Create a network interface within a VPC subnet.

        :param      subnet: EC2NetworkSubnet instance
        :type       subnet: :class:`EC2NetworkSubnet`

        :param      name:  Optional name of the interface
        :type       name:  ``str``

        :param      description:  Optional description of the network interface
        :type       description:  ``str``

        :param      private_ip_address: Optional address to assign as the
                                        primary private IP address of the
                                        interface. If one is not provided then
                                        Amazon will automatically auto-assign
                                        an available IP. EC2 allows assignment
                                        of multiple IPs, but this will be
                                        the primary.
        :type       private_ip_address: ``str``

        :return:    EC2NetworkInterface instance
        :rtype:     :class `EC2NetworkInterface`
        """
        params = {'Action': 'CreateNetworkInterface', 'SubnetId': subnet.id}
        if description:
            params['Description'] = description
        if private_ip_address:
            params['PrivateIpAddress'] = private_ip_address
        response = self.connection.request(self.path, params=params).object
        element = response.findall(fixxpath(xpath='networkInterface', namespace=NAMESPACE))[0]
        interface = self._to_interface(element, name)
        if name and self.ex_create_tags(interface, {'Name': name}):
            interface.extra['tags']['Name'] = name
        return interface

    def ex_delete_network_interface(self, network_interface):
        """
        Deletes a network interface.

        :param      network_interface: EC2NetworkInterface instance
        :type       network_interface: :class:`EC2NetworkInterface`

        :rtype:     ``bool``
        """
        params = {'Action': 'DeleteNetworkInterface', 'NetworkInterfaceId': network_interface.id}
        res = self.connection.request(self.path, params=params).object
        return self._get_boolean(res)

    def ex_attach_network_interface_to_node(self, network_interface, node, device_index):
        """
        Attach a network interface to an instance.

        :param      network_interface: EC2NetworkInterface instance
        :type       network_interface: :class:`EC2NetworkInterface`

        :param      node: Node instance
        :type       node: :class:`Node`

        :param      device_index: The interface device index
        :type       device_index: ``int``

        :return:    String representation of the attachment id.
                    This is required to detach the interface.
        :rtype:     ``str``
        """
        params = {'Action': 'AttachNetworkInterface', 'NetworkInterfaceId': network_interface.id, 'InstanceId': node.id, 'DeviceIndex': device_index}
        response = self.connection.request(self.path, params=params).object
        attachment_id = findattr(element=response, xpath='attachmentId', namespace=NAMESPACE)
        return attachment_id

    def ex_detach_network_interface(self, attachment_id, force=False):
        """
        Detach a network interface from an instance.

        :param      attachment_id: The attachment ID associated with the
                                   interface
        :type       attachment_id: ``str``

        :param      force: Forces the detachment.
        :type       force: ``bool``

        :return:    ``True`` on successful detachment, ``False`` otherwise.
        :rtype:     ``bool``
        """
        params = {'Action': 'DetachNetworkInterface', 'AttachmentId': attachment_id}
        if force:
            params['Force'] = True
        res = self.connection.request(self.path, params=params).object
        return self._get_boolean(res)

    def ex_modify_instance_attribute(self, node, attributes):
        """
        Modify node attributes.
        A list of valid attributes can be found at http://goo.gl/gxcj8

        :param      node: Node instance
        :type       node: :class:`Node`

        :param      attributes: Dictionary with node attributes
        :type       attributes: ``dict``

        :return: True on success, False otherwise.
        :rtype: ``bool``
        """
        attributes = attributes or {}
        attributes.update({'InstanceId': node.id})
        params = {'Action': 'ModifyInstanceAttribute'}
        params.update(attributes)
        res = self.connection.request(self.path, params=params.copy()).object
        return self._get_boolean(res)

    def ex_modify_snapshot_attribute(self, snapshot, attributes):
        """
        Modify Snapshot attributes.

        :param      snapshot: VolumeSnapshot instance
        :type       snanpshot: :class:`VolumeSnapshot`

        :param      attributes: Dictionary with snapshot attributes
        :type       attributes: ``dict``

        :return: True on success, False otherwise.
        :rtype: ``bool``
        """
        attributes = attributes or {}
        attributes.update({'SnapshotId': snapshot.id})
        params = {'Action': 'ModifySnapshotAttribute'}
        params.update(attributes)
        res = self.connection.request(self.path, params=params.copy()).object
        return self._get_boolean(res)

    def ex_modify_image_attribute(self, image, attributes):
        """
        Modifies image attributes.

        :param      image: NodeImage instance
        :type       image: :class:`NodeImage`

        :param      attributes: A dictionary with node attributes
        :type       attributes: ``dict``

        :return: True on success, False otherwise.
        :rtype: ``bool``
        """
        attributes = attributes or {}
        attributes.update({'ImageId': image.id})
        params = {'Action': 'ModifyImageAttribute'}
        params.update(attributes)
        res = self.connection.request(self.path, params=params.copy()).object
        return self._get_boolean(res)

    def ex_change_node_size(self, node, new_size):
        """
        Change the node size.
        Note: Node must be turned of before changing the size.

        :param      node: Node instance
        :type       node: :class:`Node`

        :param      new_size: NodeSize instance
        :type       new_size: :class:`NodeSize`

        :return: True on success, False otherwise.
        :rtype: ``bool``
        """
        if 'instancetype' in node.extra:
            current_instance_type = node.extra['instancetype']
            if current_instance_type == new_size.id:
                raise ValueError('New instance size is the same as' + 'the current one')
        attributes = {'InstanceType.Value': new_size.id}
        return self.ex_modify_instance_attribute(node=node, attributes=attributes)

    def ex_start_node(self, node):
        return self.start_node(node=node)

    def ex_stop_node(self, node):
        return self.stop_node(node=node)

    def ex_get_console_output(self, node):
        """
        Gets console output for the node.

        :param      node: Node which should be used
        :type       node: :class:`Node`

        :return:    A dictionary with the following keys:
                    - instance_id (``str``)
                    - timestamp (``datetime.datetime``) - last output timestamp
                    - output (``str``) - console output
        :rtype:     ``dict``
        """
        params = {'Action': 'GetConsoleOutput', 'InstanceId': node.id}
        response = self.connection.request(self.path, params=params).object
        timestamp = findattr(element=response, xpath='timestamp', namespace=NAMESPACE)
        encoded_string = findattr(element=response, xpath='output', namespace=NAMESPACE)
        timestamp = parse_date(timestamp)
        if encoded_string:
            output = base64.b64decode(b(encoded_string)).decode('utf-8')
        else:
            output = None
        return {'instance_id': node.id, 'timestamp': timestamp, 'output': output}

    def ex_list_reserved_nodes(self):
        """
        Lists all reserved instances/nodes which can be purchased from Amazon
        for one or three year terms. Reservations are made at a region level
        and reduce the hourly charge for instances.

        More information can be found at http://goo.gl/ulXCC7.

        :rtype: ``list`` of :class:`.EC2ReservedNode`
        """
        params = {'Action': 'DescribeReservedInstances'}
        response = self.connection.request(self.path, params=params).object
        return self._to_reserved_nodes(response, 'reservedInstancesSet/item')

    def ex_get_limits(self):
        """
        Retrieve account resource limits.

        :rtype: ``dict``
        """
        attributes = ['max-instances', 'max-elastic-ips', 'vpc-max-elastic-ips']
        params = {}
        params['Action'] = 'DescribeAccountAttributes'
        for index, attribute in enumerate(attributes):
            params['AttributeName.%s' % index] = attribute
        response = self.connection.request(self.path, params=params)
        data = response.object
        elems = data.findall(fixxpath(xpath='accountAttributeSet/item', namespace=NAMESPACE))
        result = {'resource': {}}
        for elem in elems:
            name = findtext(element=elem, xpath='attributeName', namespace=NAMESPACE)
            value = findtext(element=elem, xpath='attributeValueSet/item/attributeValue', namespace=NAMESPACE)
            result['resource'][name] = int(value)
        return result

    def ex_list_keypairs(self):
        """
        Lists all the keypair names and fingerprints.

        :rtype: ``list`` of ``dict``
        """
        warnings.warn('This method has been deprecated in favor of list_key_pairs method')
        key_pairs = self.list_key_pairs()
        result = []
        for key_pair in key_pairs:
            item = {'keyName': key_pair.name, 'keyFingerprint': key_pair.fingerprint}
            result.append(item)
        return result

    def ex_describe_all_keypairs(self):
        """
        Returns names for all the available key pairs.

        @note: This is a non-standard extension API, and only works for EC2.

        :rtype: ``list`` of ``str``
        """
        names = [key_pair.name for key_pair in self.list_key_pairs()]
        return names

    def ex_describe_keypairs(self, name):
        """
        Here for backward compatibility.
        """
        return self.ex_describe_keypair(name=name)

    def ex_describe_keypair(self, name):
        """
        Describes a keypair by name.

        @note: This is a non-standard extension API, and only works for EC2.

        :param      name: The name of the keypair to describe.
        :type       name: ``str``

        :rtype: ``dict``
        """
        params = {'Action': 'DescribeKeyPairs', 'KeyName.1': name}
        response = self.connection.request(self.path, params=params).object
        key_name = findattr(element=response, xpath='keySet/item/keyName', namespace=NAMESPACE)
        fingerprint = findattr(element=response, xpath='keySet/item/keyFingerprint', namespace=NAMESPACE).strip()
        return {'keyName': key_name, 'keyFingerprint': fingerprint}

    def ex_create_keypair(self, name):
        """
        Creates a new keypair

        @note: This is a non-standard extension API, and only works for EC2.

        :param      name: The name of the keypair to Create. This must be
            unique, otherwise an InvalidKeyPair.Duplicate exception is raised.
        :type       name: ``str``

        :rtype: ``dict``
        """
        warnings.warn('This method has been deprecated in favor of create_key_pair method')
        key_pair = self.create_key_pair(name=name)
        result = {'keyMaterial': key_pair.private_key, 'keyFingerprint': key_pair.fingerprint}
        return result

    def ex_delete_keypair(self, keypair):
        """
        Deletes a key pair by name.

        @note: This is a non-standard extension API, and only works with EC2.

        :param      keypair: The name of the keypair to delete.
        :type       keypair: ``str``

        :rtype: ``bool``
        """
        warnings.warn('This method has been deprecated in favor of delete_key_pair method')
        keypair = KeyPair(name=keypair, public_key=None, fingerprint=None, driver=self)
        return self.delete_key_pair(keypair)

    def ex_import_keypair_from_string(self, name, key_material):
        """
        Imports a new public key where the public key is passed in as a string.

        @note: This is a non-standard extension API, and only works for EC2.

        :param      name: The name of the public key to import. This must be
         unique, otherwise an InvalidKeyPair.Duplicate exception is raised.
        :type       name: ``str``

        :param     key_material: The contents of a public key file.
        :type      key_material: ``str``

        :rtype: ``dict``
        """
        warnings.warn('This method has been deprecated in favor of import_key_pair_from_string method')
        key_pair = self.import_key_pair_from_string(name=name, key_material=key_material)
        result = {'keyName': key_pair.name, 'keyFingerprint': key_pair.fingerprint}
        return result

    def ex_import_keypair(self, name, keyfile):
        """
        Imports a new public key where the public key is passed via a filename.

        @note: This is a non-standard extension API, and only works for EC2.

        :param      name: The name of the public key to import. This must be
                          unique, otherwise an InvalidKeyPair. Duplicate
                          exception is raised.
        :type       name: ``str``

        :param     keyfile: The filename with the path of the public key
                            to import.
        :type      keyfile: ``str``

        :rtype: ``dict``
        """
        warnings.warn('This method has been deprecated in favor of import_key_pair_from_file method')
        key_pair = self.import_key_pair_from_file(name=name, key_file_path=keyfile)
        result = {'keyName': key_pair.name, 'keyFingerprint': key_pair.fingerprint}
        return result

    def ex_find_or_import_keypair_by_key_material(self, pubkey):
        """
        Given a public key, look it up in the EC2 KeyPair database. If it
        exists, return any information we have about it. Otherwise, create it.

        Keys that are created are named based on their comment and fingerprint.

        :rtype: ``dict``
        """
        key_fingerprint = get_pubkey_ssh2_fingerprint(pubkey)
        key_comment = get_pubkey_comment(pubkey, default='unnamed')
        key_name = '{}-{}'.format(key_comment, key_fingerprint)
        key_pairs = self.list_key_pairs()
        key_pairs = [key_pair for key_pair in key_pairs if key_pair.fingerprint == key_fingerprint]
        if len(key_pairs) >= 1:
            key_pair = key_pairs[0]
            result = {'keyName': key_pair.name, 'keyFingerprint': key_pair.fingerprint}
        else:
            result = self.ex_import_keypair_from_string(key_name, pubkey)
        return result

    def ex_list_internet_gateways(self, gateway_ids=None, filters=None):
        """
        Describes available Internet gateways and whether or not they are
        attached to a VPC. These are required for VPC nodes to communicate
        over the Internet.

        :param      gateway_ids: Returns only Internet gateways matching the
                                 provided Internet gateway IDs. If not
                                 specified, a list of all the Internet
                                 gateways in the corresponding region is
                                 returned.
        :type       gateway_ids: ``list``

        :param      filters: The filters so the list returned includes
                             information for certain gateways only.
        :type       filters: ``dict``

        :rtype: ``list`` of :class:`.VPCInternetGateway`
        """
        params = {'Action': 'DescribeInternetGateways'}
        if gateway_ids:
            params.update(self._pathlist('InternetGatewayId', gateway_ids))
        if filters:
            params.update(self._build_filters(filters))
        response = self.connection.request(self.path, params=params).object
        return self._to_internet_gateways(response, 'internetGatewaySet/item')

    def ex_create_internet_gateway(self, name=None):
        """
        Delete a VPC Internet gateway

        :rtype:     ``bool``
        """
        params = {'Action': 'CreateInternetGateway'}
        resp = self.connection.request(self.path, params=params).object
        element = resp.findall(fixxpath(xpath='internetGateway', namespace=NAMESPACE))
        gateway = self._to_internet_gateway(element[0], name)
        if name and self.ex_create_tags(gateway, {'Name': name}):
            gateway.extra['tags']['Name'] = name
        return gateway

    def ex_delete_internet_gateway(self, gateway):
        """
        Deletes a VPC Internet gateway.

        :param      gateway: The gateway to delete
        :type       gateway: :class:`.VPCInternetGateway`

        :rtype:     ``bool``
        """
        params = {'Action': 'DeleteInternetGateway', 'InternetGatewayId': gateway.id}
        res = self.connection.request(self.path, params=params).object
        return self._get_boolean(res)

    def ex_attach_internet_gateway(self, gateway, network):
        """
        Attach an Internet gateway to a VPC

        :param      gateway: The gateway to attach
        :type       gateway: :class:`.VPCInternetGateway`

        :param      network: The VPC network to attach to
        :type       network: :class:`.EC2Network`

        :rtype:     ``bool``
        """
        params = {'Action': 'AttachInternetGateway', 'InternetGatewayId': gateway.id, 'VpcId': network.id}
        res = self.connection.request(self.path, params=params).object
        return self._get_boolean(res)

    def ex_detach_internet_gateway(self, gateway, network):
        """
        Detaches an Internet gateway from a VPC.

        :param      gateway: The gateway to detach
        :type       gateway: :class:`.VPCInternetGateway`

        :param      network: The VPC network to detach from
        :type       network: :class:`.EC2Network`

        :rtype:     ``bool``
        """
        params = {'Action': 'DetachInternetGateway', 'InternetGatewayId': gateway.id, 'VpcId': network.id}
        res = self.connection.request(self.path, params=params).object
        return self._get_boolean(res)

    def ex_list_route_tables(self, route_table_ids=None, filters=None):
        """
        Describes one or more of a VPC's route tables.
        These are used to determine where network traffic is directed.

        :param      route_table_ids: Returns only route tables matching the
                                provided route table IDs. If not specified,
                                a list of all the route tables in the
                                corresponding region is returned.
        :type       route_table_ids: ``list``

        :param      filters: The filters so that the list returned includes
                             information for certain route tables only.
        :type       filters: ``dict``

        :rtype: ``list`` of :class:`.EC2RouteTable`
        """
        params = {'Action': 'DescribeRouteTables'}
        if route_table_ids:
            params.update(self._pathlist('RouteTableId', route_table_ids))
        if filters:
            params.update(self._build_filters(filters))
        response = self.connection.request(self.path, params=params)
        return self._to_route_tables(response.object)

    def ex_create_route_table(self, network, name=None):
        """
        Creates a route table within a VPC.

        :param      vpc_id: The VPC that the subnet should be created in.
        :type       vpc_id: :class:`.EC2Network`

        :rtype:     :class: `.EC2RouteTable`
        """
        params = {'Action': 'CreateRouteTable', 'VpcId': network.id}
        response = self.connection.request(self.path, params=params).object
        element = response.findall(fixxpath(xpath='routeTable', namespace=NAMESPACE))[0]
        route_table = self._to_route_table(element, name=name)
        if name and self.ex_create_tags(route_table, {'Name': name}):
            route_table.extra['tags']['Name'] = name
        return route_table

    def ex_delete_route_table(self, route_table):
        """
        Deletes a VPC route table.

        :param      route_table: The route table to delete.
        :type       route_table: :class:`.EC2RouteTable`

        :rtype:     ``bool``
        """
        params = {'Action': 'DeleteRouteTable', 'RouteTableId': route_table.id}
        res = self.connection.request(self.path, params=params).object
        return self._get_boolean(res)

    def ex_associate_route_table(self, route_table, subnet):
        """
        Associates a route table with a subnet within a VPC.

        Note: A route table can be associated with multiple subnets.

        :param      route_table: The route table to associate.
        :type       route_table: :class:`.EC2RouteTable`

        :param      subnet: The subnet to associate with.
        :type       subnet: :class:`.EC2Subnet`

        :return:    Route table association ID.
        :rtype:     ``str``
        """
        params = {'Action': 'AssociateRouteTable', 'RouteTableId': route_table.id, 'SubnetId': subnet.id}
        result = self.connection.request(self.path, params=params).object
        association_id = findtext(element=result, xpath='associationId', namespace=NAMESPACE)
        return association_id

    def ex_dissociate_route_table(self, subnet_association):
        """
        Dissociates a subnet from a route table.

        :param      subnet_association: The subnet association object or
                                        subnet association ID.
        :type       subnet_association: :class:`.EC2SubnetAssociation` or
                                        ``str``

        :rtype:     ``bool``
        """
        if isinstance(subnet_association, EC2SubnetAssociation):
            subnet_association_id = subnet_association.id
        else:
            subnet_association_id = subnet_association
        params = {'Action': 'DisassociateRouteTable', 'AssociationId': subnet_association_id}
        res = self.connection.request(self.path, params=params).object
        return self._get_boolean(res)

    def ex_replace_route_table_association(self, subnet_association, route_table):
        """
        Changes the route table associated with a given subnet in a VPC.

        Note: This method can be used to change which table is the main route
              table in the VPC (Specify the main route table's association ID
              and the route table to be the new main route table).

        :param      subnet_association: The subnet association object or
                                        subnet association ID.
        :type       subnet_association: :class:`.EC2SubnetAssociation` or
                                        ``str``

        :param      route_table: The new route table to associate.
        :type       route_table: :class:`.EC2RouteTable`

        :return:    A new route table association ID.
        :rtype:     ``str``
        """
        if isinstance(subnet_association, EC2SubnetAssociation):
            subnet_association_id = subnet_association.id
        else:
            subnet_association_id = subnet_association
        params = {'Action': 'ReplaceRouteTableAssociation', 'AssociationId': subnet_association_id, 'RouteTableId': route_table.id}
        result = self.connection.request(self.path, params=params).object
        new_association_id = findtext(element=result, xpath='newAssociationId', namespace=NAMESPACE)
        return new_association_id

    def ex_create_route(self, route_table, cidr, internet_gateway=None, node=None, network_interface=None, vpc_peering_connection=None):
        """
        Creates a route entry in the route table.

        :param      route_table: The route table to create the route in.
        :type       route_table: :class:`.EC2RouteTable`

        :param      cidr: The CIDR block used for the destination match.
        :type       cidr: ``str``

        :param      internet_gateway: The Internet gateway to route
                                      traffic through.
        :type       internet_gateway: :class:`.VPCInternetGateway`

        :param      node: The NAT instance to route traffic through.
        :type       node: :class:`Node`

        :param      network_interface: The network interface of the node
                                       to route traffic through.
        :type       network_interface: :class:`.EC2NetworkInterface`

        :param      vpc_peering_connection: The VPC peering connection.
        :type       vpc_peering_connection: :class:`.VPCPeeringConnection`

        :rtype:     ``bool``

        Note: You must specify one of the following: internet_gateway,
              node, network_interface, vpc_peering_connection.
        """
        params = {'Action': 'CreateRoute', 'RouteTableId': route_table.id, 'DestinationCidrBlock': cidr}
        if internet_gateway:
            params['GatewayId'] = internet_gateway.id
        if node:
            params['InstanceId'] = node.id
        if network_interface:
            params['NetworkInterfaceId'] = network_interface.id
        if vpc_peering_connection:
            params['VpcPeeringConnectionId'] = vpc_peering_connection.id
        res = self.connection.request(self.path, params=params).object
        return self._get_boolean(res)

    def ex_delete_route(self, route_table, cidr):
        """
        Deletes a route entry from the route table.

        :param      route_table: The route table to delete the route from.
        :type       route_table: :class:`.EC2RouteTable`

        :param      cidr: The CIDR block used for the destination match.
        :type       cidr: ``str``

        :rtype:     ``bool``
        """
        params = {'Action': 'DeleteRoute', 'RouteTableId': route_table.id, 'DestinationCidrBlock': cidr}
        res = self.connection.request(self.path, params=params).object
        return self._get_boolean(res)

    def ex_replace_route(self, route_table, cidr, internet_gateway=None, node=None, network_interface=None, vpc_peering_connection=None):
        """
        Replaces an existing route entry within a route table in a VPC.

        :param      route_table: The route table to replace the route in.
        :type       route_table: :class:`.EC2RouteTable`

        :param      cidr: The CIDR block used for the destination match.
        :type       cidr: ``str``

        :param      internet_gateway: The new internet gateway to route
                                       traffic through.
        :type       internet_gateway: :class:`.VPCInternetGateway`

        :param      node: The new NAT instance to route traffic through.
        :type       node: :class:`Node`

        :param      network_interface: The new network interface of the node
                                       to route traffic through.
        :type       network_interface: :class:`.EC2NetworkInterface`

        :param      vpc_peering_connection: The new VPC peering connection.
        :type       vpc_peering_connection: :class:`.VPCPeeringConnection`

        :rtype:     ``bool``

        Note: You must specify one of the following: internet_gateway,
              node, network_interface, vpc_peering_connection.
        """
        params = {'Action': 'ReplaceRoute', 'RouteTableId': route_table.id, 'DestinationCidrBlock': cidr}
        if internet_gateway:
            params['GatewayId'] = internet_gateway.id
        if node:
            params['InstanceId'] = node.id
        if network_interface:
            params['NetworkInterfaceId'] = network_interface.id
        if vpc_peering_connection:
            params['VpcPeeringConnectionId'] = vpc_peering_connection.id
        res = self.connection.request(self.path, params=params).object
        return self._get_boolean(res)

    def ex_modify_volume(self, volume, parameters):
        """
        Modify volume parameters.
        A list of valid parameters can be found at https://goo.gl/N0rPEQ

        :param      volume: Volume instance
        :type       volume: :class:`Volume`

        :param      parameters: Dictionary with updated volume parameters
        :type       parameters: ``dict``

        :return: Volume modification status object
        :rtype: :class:`VolumeModification
        """
        parameters = parameters or {}
        volume_type = parameters.get('VolumeType')
        if volume_type and volume_type not in VALID_VOLUME_TYPES:
            raise ValueError('Invalid volume type specified: %s' % volume_type)
        parameters.update({'Action': 'ModifyVolume', 'VolumeId': volume.id})
        response = self.connection.request(self.path, params=parameters.copy()).object
        return self._to_volume_modification(response.findall(fixxpath(xpath='volumeModification', namespace=NAMESPACE))[0])

    def ex_describe_volumes_modifications(self, dry_run=False, volume_ids=None, filters=None):
        """
        Describes one or more of your volume modifications.

        :param      dry_run: dry_run
        :type       dry_run: ``bool``

        :param      volume_ids: The volume_ids so that the response includes
                             information for only said volumes
        :type       volume_ids: ``dict``

        :param      filters: The filters so that the response includes
                             information for only certain volumes
        :type       filters: ``dict``

        :return:  List of volume modification status objects
        :rtype:   ``list`` of :class:`VolumeModification
        """
        params = {'Action': 'DescribeVolumesModifications'}
        if dry_run:
            params.update({'DryRun': dry_run})
        if volume_ids:
            params.update(self._pathlist('VolumeId', volume_ids))
        if filters:
            params.update(self._build_filters(filters))
        response = self.connection.request(self.path, params=params).object
        return self._to_volume_modifications(response)

    def _ex_connection_class_kwargs(self):
        kwargs = super()._ex_connection_class_kwargs()
        if hasattr(self, 'token') and self.token is not None:
            kwargs['token'] = self.token
            kwargs['signature_version'] = '4'
        else:
            kwargs['signature_version'] = self.signature_version
        return kwargs

    def _to_nodes(self, object, xpath):
        return [self._to_node(el) for el in object.findall(fixxpath(xpath=xpath, namespace=NAMESPACE))]

    def _to_node(self, element):
        try:
            state = self.NODE_STATE_MAP[findattr(element=element, xpath='instanceState/name', namespace=NAMESPACE)]
        except KeyError:
            state = NodeState.UNKNOWN
        created = parse_date_allow_empty(findtext(element=element, xpath='launchTime', namespace=NAMESPACE))
        instance_id = findtext(element=element, xpath='instanceId', namespace=NAMESPACE)
        public_ip = findtext(element=element, xpath='ipAddress', namespace=NAMESPACE)
        public_ips = [public_ip] if public_ip else []
        private_ip = findtext(element=element, xpath='privateIpAddress', namespace=NAMESPACE)
        private_ips = [private_ip] if private_ip else []
        product_codes = []
        for p in findall(element=element, xpath='productCodesSet/item/productCode', namespace=NAMESPACE):
            product_codes.append(p)
        tags = self._get_resource_tags(element)
        name = tags.get('Name', instance_id)
        extra = self._get_extra_dict(element, RESOURCE_EXTRA_ATTRIBUTES_MAP['node'])
        extra['block_device_mapping'] = self._to_instance_device_mappings(element)
        extra['groups'] = self._get_security_groups(element)
        extra['network_interfaces'] = self._to_interfaces(element)
        extra['product_codes'] = product_codes
        extra['tags'] = tags
        return Node(id=instance_id, name=name, state=state, public_ips=public_ips, private_ips=private_ips, driver=self.connection.driver, created_at=created, extra=extra)

    def _to_images(self, object):
        return [self._to_image(el) for el in object.findall(fixxpath(xpath='imagesSet/item', namespace=NAMESPACE))]

    def _to_image(self, element):
        id = findtext(element=element, xpath='imageId', namespace=NAMESPACE)
        name = findtext(element=element, xpath='name', namespace=NAMESPACE)
        block_device_mapping = self._to_device_mappings(element)
        billing_products = []
        for p in findall(element=element, xpath='billingProducts/item/billingProduct', namespace=NAMESPACE):
            billing_products.append(p.text)
        tags = self._get_resource_tags(element)
        extra = self._get_extra_dict(element, RESOURCE_EXTRA_ATTRIBUTES_MAP['image'])
        extra['tags'] = tags
        extra['block_device_mapping'] = block_device_mapping
        extra['billing_products'] = billing_products
        return NodeImage(id=id, name=name, driver=self, extra=extra)

    def _to_volume(self, element, name=None):
        """
        Parse the XML element and return a StorageVolume object.

        :param      name: An optional name for the volume. If not provided
                          then either tag with a key "Name" or volume ID
                          will be used (which ever is available first in that
                          order).
        :type       name: ``str``

        :rtype:     :class:`StorageVolume`
        """
        volId = findtext(element=element, xpath='volumeId', namespace=NAMESPACE)
        size = findtext(element=element, xpath='size', namespace=NAMESPACE)
        raw_state = findtext(element=element, xpath='status', namespace=NAMESPACE)
        state = self.VOLUME_STATE_MAP.get(raw_state, StorageVolumeState.UNKNOWN)
        tags = self._get_resource_tags(element)
        name = name if name else tags.get('Name', volId)
        extra = self._get_extra_dict(element, RESOURCE_EXTRA_ATTRIBUTES_MAP['volume'])
        extra['tags'] = tags
        return StorageVolume(id=volId, name=name, size=int(size), driver=self, state=state, extra=extra)

    def _to_volume_modifications(self, object):
        return [self._to_volume_modification(el) for el in object.findall(fixxpath(xpath='volumeModificationSet/item', namespace=NAMESPACE))]

    def _to_volume_modification(self, element):
        """
        Parse the XML element and return a StorageVolume object.

        :rtype:     :class:`EC2VolumeModification`
        """
        params = self._get_extra_dict(element, VOLUME_MODIFICATION_ATTRIBUTE_MAP)
        return EC2VolumeModification(**params)

    def _to_snapshots(self, response):
        return [self._to_snapshot(el) for el in response.findall(fixxpath(xpath='snapshotSet/item', namespace=NAMESPACE))]

    def _to_snapshot(self, element, name=None):
        snapId = findtext(element=element, xpath='snapshotId', namespace=NAMESPACE)
        size = findtext(element=element, xpath='volumeSize', namespace=NAMESPACE)
        created = parse_date_allow_empty(findtext(element=element, xpath='startTime', namespace=NAMESPACE))
        tags = self._get_resource_tags(element)
        name = name if name else tags.get('Name', snapId)
        extra = self._get_extra_dict(element, RESOURCE_EXTRA_ATTRIBUTES_MAP['snapshot'])
        extra['tags'] = tags
        extra['name'] = name
        state = self.SNAPSHOT_STATE_MAP.get(extra['state'], VolumeSnapshotState.UNKNOWN)
        return VolumeSnapshot(snapId, size=int(size), driver=self, extra=extra, created=created, state=state, name=name)

    def _to_import_snapshot_task(self, element):
        status = findtext(element=element, xpath='importSnapshotTaskSet/item/snapshotTaskDetail/status', namespace=NAMESPACE)
        if status != 'completed':
            snapshotId = None
        else:
            xpath = 'importSnapshotTaskSet/item/snapshotTaskDetail/snapshotId'
            snapshotId = findtext(element=element, xpath=xpath, namespace=NAMESPACE)
        return EC2ImportSnapshotTask(status, snapshotId=snapshotId)

    def _to_key_pairs(self, elems):
        key_pairs = [self._to_key_pair(elem=elem) for elem in elems]
        return key_pairs

    def _to_key_pair(self, elem):
        name = findtext(element=elem, xpath='keyName', namespace=NAMESPACE)
        fingerprint = findtext(element=elem, xpath='keyFingerprint', namespace=NAMESPACE).strip()
        private_key = findtext(element=elem, xpath='keyMaterial', namespace=NAMESPACE)
        key_pair = KeyPair(name=name, public_key=None, fingerprint=fingerprint, private_key=private_key, driver=self)
        return key_pair

    def _to_security_groups(self, response):
        return [self._to_security_group(el) for el in response.findall(fixxpath(xpath='securityGroupInfo/item', namespace=NAMESPACE))]

    def _to_security_group(self, element):
        sg_id = findtext(element=element, xpath='groupId', namespace=NAMESPACE)
        name = findtext(element=element, xpath='groupName', namespace=NAMESPACE)
        tags = self._get_resource_tags(element)
        extra = self._get_extra_dict(element, RESOURCE_EXTRA_ATTRIBUTES_MAP['security_group'])
        extra['tags'] = tags
        ingress_rules = self._to_security_group_rules(element, 'ipPermissions/item')
        egress_rules = self._to_security_group_rules(element, 'ipPermissionsEgress/item')
        return EC2SecurityGroup(sg_id, name, ingress_rules, egress_rules, extra=extra)

    def _to_security_group_rules(self, element, xpath):
        return [self._to_security_group_rule(el) for el in element.findall(fixxpath(xpath=xpath, namespace=NAMESPACE))]

    def _to_security_group_rule(self, element):
        """
        Parse the XML element and return a SecurityGroup object.

        :rtype:     :class:`EC2SecurityGroup`
        """
        rule = {}
        rule['protocol'] = findtext(element=element, xpath='ipProtocol', namespace=NAMESPACE)
        rule['from_port'] = findtext(element=element, xpath='fromPort', namespace=NAMESPACE)
        rule['to_port'] = findtext(element=element, xpath='toPort', namespace=NAMESPACE)
        elements = element.findall(fixxpath(xpath='groups/item', namespace=NAMESPACE))
        rule['group_pairs'] = []
        for element in elements:
            item = {'user_id': findtext(element=element, xpath='userId', namespace=NAMESPACE), 'group_id': findtext(element=element, xpath='groupId', namespace=NAMESPACE), 'group_name': findtext(element=element, xpath='groupName', namespace=NAMESPACE)}
            rule['group_pairs'].append(item)
        elements = element.findall(fixxpath(xpath='ipRanges/item', namespace=NAMESPACE))
        rule['cidr_ips'] = [findtext(element=element, xpath='cidrIp', namespace=NAMESPACE) for element in elements]
        return rule

    def _to_networks(self, response):
        return [self._to_network(el) for el in response.findall(fixxpath(xpath='vpcSet/item', namespace=NAMESPACE))]

    def _to_network(self, element, name=None):
        vpc_id = findtext(element=element, xpath='vpcId', namespace=NAMESPACE)
        tags = self._get_resource_tags(element)
        name = name if name else tags.get('Name', vpc_id)
        cidr_block = findtext(element=element, xpath='cidrBlock', namespace=NAMESPACE)
        extra = self._get_extra_dict(element, RESOURCE_EXTRA_ATTRIBUTES_MAP['network'])
        extra['tags'] = tags
        return EC2Network(vpc_id, name, cidr_block, extra=extra)

    def _to_addresses(self, response, only_associated):
        """
        Builds a list of dictionaries containing elastic IP properties.

        :param    only_associated: If true, return only those addresses
                                   that are associated with an instance.
                                   If false, return all addresses.
        :type     only_associated: ``bool``

        :rtype:   ``list`` of :class:`ElasticIP`
        """
        addresses = []
        for el in response.findall(fixxpath(xpath='addressesSet/item', namespace=NAMESPACE)):
            addr = self._to_address(el, only_associated)
            if addr is not None:
                addresses.append(addr)
        return addresses

    def _to_address(self, element, only_associated):
        instance_id = findtext(element=element, xpath='instanceId', namespace=NAMESPACE)
        public_ip = findtext(element=element, xpath='publicIp', namespace=NAMESPACE)
        domain = findtext(element=element, xpath='domain', namespace=NAMESPACE)
        extra = self._get_extra_dict(element, RESOURCE_EXTRA_ATTRIBUTES_MAP['elastic_ip'])
        if only_associated and (not instance_id):
            return None
        return ElasticIP(public_ip, domain, instance_id, extra=extra)

    def _to_placement_groups(self, response):
        return [self._to_placement_group(el) for el in response.findall(fixxpath(xpath='placementGroupSet/item', namespace=NAMESPACE))]

    def _to_placement_group(self, element):
        name = findtext(element=element, xpath='groupName', namespace=NAMESPACE)
        state = findtext(element=element, xpath='state', namespace=NAMESPACE)
        strategy = findtext(element=element, xpath='strategy', namespace=NAMESPACE)
        return EC2PlacementGroup(name, state, strategy)

    def _to_subnets(self, response):
        return [self._to_subnet(el) for el in response.findall(fixxpath(xpath='subnetSet/item', namespace=NAMESPACE))]

    def _to_subnet(self, element, name=None):
        subnet_id = findtext(element=element, xpath='subnetId', namespace=NAMESPACE)
        tags = self._get_resource_tags(element)
        name = name if name else tags.get('Name', subnet_id)
        state = findtext(element=element, xpath='state', namespace=NAMESPACE)
        extra = self._get_extra_dict(element, RESOURCE_EXTRA_ATTRIBUTES_MAP['subnet'])
        extra['tags'] = tags
        return EC2NetworkSubnet(subnet_id, name, state, extra=extra)

    def _to_interfaces(self, response):
        return [self._to_interface(el) for el in response.findall(fixxpath(xpath='networkInterfaceSet/item', namespace=NAMESPACE))]

    def _to_interface(self, element, name=None):
        """
        Parse the XML element and return an EC2NetworkInterface object.

        :param      name: An optional name for the interface. If not provided
                          then either tag with a key "Name" or the interface ID
                          will be used (whichever is available first in that
                          order).
        :type       name: ``str``

        :rtype:     :class: `EC2NetworkInterface`
        """
        interface_id = findtext(element=element, xpath='networkInterfaceId', namespace=NAMESPACE)
        state = findtext(element=element, xpath='status', namespace=NAMESPACE)
        tags = self._get_resource_tags(element)
        name = name if name else tags.get('Name', interface_id)
        groups = self._get_security_groups(element)
        priv_ips = []
        for item in findall(element=element, xpath='privateIpAddressesSet/item', namespace=NAMESPACE):
            priv_ips.append({'private_ip': findtext(element=item, xpath='privateIpAddress', namespace=NAMESPACE), 'private_dns': findtext(element=item, xpath='privateDnsName', namespace=NAMESPACE), 'primary': findtext(element=item, xpath='primary', namespace=NAMESPACE)})
        attributes_map = RESOURCE_EXTRA_ATTRIBUTES_MAP['network_interface_attachment']
        attachment = self._get_extra_dict(element, attributes_map)
        attributes_map = RESOURCE_EXTRA_ATTRIBUTES_MAP['network_interface']
        extra = self._get_extra_dict(element, attributes_map)
        extra['tags'] = tags
        extra['attachment'] = attachment
        extra['private_ips'] = priv_ips
        extra['groups'] = groups
        return EC2NetworkInterface(interface_id, name, state, extra=extra)

    def _to_reserved_nodes(self, object, xpath):
        return [self._to_reserved_node(el) for el in object.findall(fixxpath(xpath=xpath, namespace=NAMESPACE))]

    def _to_reserved_node(self, element):
        """
        Build an EC2ReservedNode object using the reserved instance properties.
        Information on these properties can be found at http://goo.gl/ulXCC7.
        """
        extra = self._get_extra_dict(element, RESOURCE_EXTRA_ATTRIBUTES_MAP['reserved_node'])
        try:
            size = [size for size in self.list_sizes() if size.id == extra['instance_type']][0]
        except IndexError:
            size = None
        return EC2ReservedNode(id=findtext(element=element, xpath='reservedInstancesId', namespace=NAMESPACE), state=findattr(element=element, xpath='state', namespace=NAMESPACE), driver=self, size=size, extra=extra)

    def _to_device_mappings(self, object):
        return [self._to_device_mapping(el) for el in object.findall(fixxpath(xpath='blockDeviceMapping/item', namespace=NAMESPACE))]

    def _to_device_mapping(self, element):
        """
        Parse the XML element and return a dictionary of device properties.
        Additional information can be found at http://goo.gl/GjWYBf.

        @note: EBS volumes do not have a virtual name. Only ephemeral
               disks use this property.
        :rtype:     ``dict``
        """
        mapping = {}
        mapping['device_name'] = findattr(element=element, xpath='deviceName', namespace=NAMESPACE)
        mapping['virtual_name'] = findattr(element=element, xpath='virtualName', namespace=NAMESPACE)
        if mapping['virtual_name'] is None:
            mapping['ebs'] = self._get_extra_dict(element, RESOURCE_EXTRA_ATTRIBUTES_MAP['ebs_volume'])
        return mapping

    def _to_instance_device_mappings(self, object):
        return [self._to_instance_device_mapping(el) for el in object.findall(fixxpath(xpath='blockDeviceMapping/item', namespace=NAMESPACE))]

    def _to_instance_device_mapping(self, element):
        """
        Parse the XML element and return a dictionary of device properties.
        Additional information can be found at https://goo.gl/OGK88a.

        :rtype:     ``dict``
        """
        mapping = {}
        mapping['device_name'] = findattr(element=element, xpath='deviceName', namespace=NAMESPACE)
        mapping['ebs'] = self._get_extra_dict(element, RESOURCE_EXTRA_ATTRIBUTES_MAP['ebs_instance_block_device'])
        return mapping

    def _to_internet_gateways(self, object, xpath):
        return [self._to_internet_gateway(el) for el in object.findall(fixxpath(xpath=xpath, namespace=NAMESPACE))]

    def _to_internet_gateway(self, element, name=None):
        id = findtext(element=element, xpath='internetGatewayId', namespace=NAMESPACE)
        vpc_id = findtext(element=element, xpath='attachmentSet/item/vpcId', namespace=NAMESPACE)
        state = findtext(element=element, xpath='attachmentSet/item/state', namespace=NAMESPACE)
        if not state:
            state = 'available'
        tags = self._get_resource_tags(element)
        name = name if name else tags.get('Name', id)
        return VPCInternetGateway(id=id, name=name, vpc_id=vpc_id, state=state, driver=self.connection.driver, extra={'tags': tags})

    def _to_route_tables(self, response):
        return [self._to_route_table(el) for el in response.findall(fixxpath(xpath='routeTableSet/item', namespace=NAMESPACE))]

    def _to_route_table(self, element, name=None):
        route_table_id = findtext(element=element, xpath='routeTableId', namespace=NAMESPACE)
        tags = self._get_resource_tags(element)
        extra = self._get_extra_dict(element, RESOURCE_EXTRA_ATTRIBUTES_MAP['route_table'])
        extra['tags'] = tags
        routes = self._to_routes(element, 'routeSet/item')
        subnet_associations = self._to_subnet_associations(element, 'associationSet/item')
        propagating_gateway_ids = []
        for el in element.findall(fixxpath(xpath='propagatingVgwSet/item', namespace=NAMESPACE)):
            propagating_gateway_ids.append(findtext(element=el, xpath='gatewayId', namespace=NAMESPACE))
        name = name if name else tags.get('Name', id)
        return EC2RouteTable(route_table_id, name, routes, subnet_associations, propagating_gateway_ids, extra=extra)

    def _to_routes(self, element, xpath):
        return [self._to_route(el) for el in element.findall(fixxpath(xpath=xpath, namespace=NAMESPACE))]

    def _to_route(self, element):
        """
        Parse the XML element and return a route object

        :rtype:     :class: `EC2Route`
        """
        destination_cidr = findtext(element=element, xpath='destinationCidrBlock', namespace=NAMESPACE)
        gateway_id = findtext(element=element, xpath='gatewayId', namespace=NAMESPACE)
        instance_id = findtext(element=element, xpath='instanceId', namespace=NAMESPACE)
        owner_id = findtext(element=element, xpath='instanceOwnerId', namespace=NAMESPACE)
        interface_id = findtext(element=element, xpath='networkInterfaceId', namespace=NAMESPACE)
        state = findtext(element=element, xpath='state', namespace=NAMESPACE)
        origin = findtext(element=element, xpath='origin', namespace=NAMESPACE)
        vpc_peering_connection_id = findtext(element=element, xpath='vpcPeeringConnectionId', namespace=NAMESPACE)
        return EC2Route(destination_cidr, gateway_id, instance_id, owner_id, interface_id, state, origin, vpc_peering_connection_id)

    def _to_subnet_associations(self, element, xpath):
        return [self._to_subnet_association(el) for el in element.findall(fixxpath(xpath=xpath, namespace=NAMESPACE))]

    def _to_subnet_association(self, element):
        """
        Parse the XML element and return a route table association object

        :rtype:     :class: `EC2SubnetAssociation`
        """
        association_id = findtext(element=element, xpath='routeTableAssociationId', namespace=NAMESPACE)
        route_table_id = findtext(element=element, xpath='routeTableId', namespace=NAMESPACE)
        subnet_id = findtext(element=element, xpath='subnetId', namespace=NAMESPACE)
        main = findtext(element=element, xpath='main', namespace=NAMESPACE)
        main = True if main else False
        return EC2SubnetAssociation(association_id, route_table_id, subnet_id, main)

    def _pathlist(self, key, arr):
        """
        Converts a key and an array of values into AWS query param format.
        """
        params = {}
        i = 0
        for value in arr:
            i += 1
            params['{}.{}'.format(key, i)] = value
        return params

    def _get_boolean(self, element):
        tag = '{{{}}}{}'.format(NAMESPACE, 'return')
        return element.findtext(tag) == 'true'

    def _get_terminate_boolean(self, element):
        status = element.findtext('.//{{{}}}{}'.format(NAMESPACE, 'name'))
        return any([term_status == status for term_status in ('shutting-down', 'terminated')])

    def _add_instance_filter(self, params, node):
        """
        Add instance filter to the provided params dictionary.
        """
        filters = {'instance-id': node.id}
        params.update(self._build_filters(filters))
        return params

    def _get_state_boolean(self, element):
        """
        Checks for the instances's state
        """
        state = findall(element=element, xpath='instancesSet/item/currentState/name', namespace=NAMESPACE)[0].text
        return state in ('stopping', 'pending', 'starting')

    def _get_extra_dict(self, element, mapping):
        """
        Extract attributes from the element based on rules provided in the
        mapping dictionary.

        :param      element: Element to parse the values from.
        :type       element: xml.etree.ElementTree.Element.

        :param      mapping: Dictionary with the extra layout
        :type       node: :class:`Node`

        :rtype: ``dict``
        """
        extra = {}
        for attribute, values in mapping.items():
            transform_func = values['transform_func']
            value = findattr(element=element, xpath=values['xpath'], namespace=NAMESPACE)
            if value is not None:
                extra[attribute] = transform_func(value)
            else:
                extra[attribute] = None
        return extra

    def _get_resource_tags(self, element):
        """
        Parse tags from the provided element and return a dictionary with
        key/value pairs.

        :rtype: ``dict``
        """
        tags = {}
        tag_set = findall(element=element, xpath='tagSet/item', namespace=NAMESPACE)
        for tag in tag_set:
            key = findtext(element=tag, xpath='key', namespace=NAMESPACE)
            value = findtext(element=tag, xpath='value', namespace=NAMESPACE)
            tags[key] = value
        return tags

    def _get_block_device_mapping_params(self, block_device_mapping):
        """
        Return a list of dictionaries with query parameters for
        a valid block device mapping.

        :param      mapping: List of dictionaries with the drive layout
        :type       mapping: ``list`` or ``dict``

        :return:    Dictionary representation of the drive mapping
        :rtype:     ``dict``
        """
        if not isinstance(block_device_mapping, (list, tuple)):
            raise AttributeError('block_device_mapping not list or tuple')
        params = {}
        for idx, mapping in enumerate(block_device_mapping):
            idx += 1
            if not isinstance(mapping, dict):
                raise AttributeError('mapping %s in block_device_mapping not a dict' % mapping)
            for k, v in mapping.items():
                if not isinstance(v, dict):
                    params['BlockDeviceMapping.%d.%s' % (idx, k)] = str(v)
                else:
                    for key, value in v.items():
                        params['BlockDeviceMapping.%d.%s.%s' % (idx, k, key)] = str(value)
        return params

    def _get_billing_product_params(self, billing_products):
        """
        Return a list of dictionaries with valid param for billing product.

        :param      billing_product: List of billing code values(str)
        :type       billing product: ``list``

        :return:    Dictionary representation of the billing product codes
        :rtype:     ``dict``
        """
        if not isinstance(billing_products, (list, tuple)):
            raise AttributeError('billing_products not list or tuple')
        params = {}
        for idx, v in enumerate(billing_products):
            idx += 1
            params['BillingProduct.%d' % idx] = str(v)
        return params

    def _get_disk_container_params(self, disk_container):
        """
        Return a list of dictionaries with query parameters for
        a valid disk container.

        :param      disk_container: List of dictionaries with
                                    disk_container details
        :type       disk_container: ``list`` or ``dict``

        :return:    Dictionary representation of the disk_container
        :rtype:     ``dict``
        """
        if not isinstance(disk_container, (list, tuple)):
            raise AttributeError('disk_container not list or tuple')
        params = {}
        for idx, content in enumerate(disk_container):
            idx += 1
            if not isinstance(content, dict):
                raise AttributeError('content %s in disk_container not a dict' % content)
            for k, v in content.items():
                if not isinstance(v, dict):
                    params['DiskContainer.%s' % k] = str(v)
                else:
                    for key, value in v.items():
                        params['DiskContainer.{}.{}'.format(k, key)] = str(value)
        return params

    def _get_client_data_params(self, client_data):
        """
        Return a dictionary with query parameters for
        a valid client data.

        :param      client_data: List of dictionaries with the disk
                                 upload details
        :type       client_data: ``dict``

        :return:    Dictionary representation of the client data
        :rtype:     ``dict``
        """
        if not isinstance(client_data, (list, tuple)):
            raise AttributeError('client_data not list or tuple')
        params = {}
        for idx, content in enumerate(client_data):
            idx += 1
            if not isinstance(content, dict):
                raise AttributeError('content %s in client_datanot a dict' % content)
            for k, v in content.items():
                params['ClientData.%s' % k] = str(v)
        return params

    def _get_common_security_group_params(self, group_id, protocol, from_port, to_port, cidr_ips, group_pairs, description=None):
        """
        Return a dictionary with common query parameters which are used when
        operating on security groups.

        :rtype: ``dict``
        """
        params = {'GroupId': group_id, 'IpPermissions.1.IpProtocol': protocol, 'IpPermissions.1.FromPort': from_port, 'IpPermissions.1.ToPort': to_port}
        if cidr_ips is not None:
            ip_ranges = {}
            for index, cidr_ip in enumerate(cidr_ips):
                index += 1
                ip_ranges['IpPermissions.1.IpRanges.%s.CidrIp' % index] = cidr_ip
                if description is not None:
                    ip_ranges['IpPermissions.1.IpRanges.%s.Description' % index] = description
            params.update(ip_ranges)
        if group_pairs is not None:
            user_groups = {}
            for index, group_pair in enumerate(group_pairs):
                index += 1
                if 'group_id' in group_pair.keys():
                    user_groups['IpPermissions.1.Groups.%s.GroupId' % index] = group_pair['group_id']
                if 'group_name' in group_pair.keys():
                    user_groups['IpPermissions.1.Groups.%s.GroupName' % index] = group_pair['group_name']
                if 'user_id' in group_pair.keys():
                    user_groups['IpPermissions.1.Groups.%s.UserId' % index] = group_pair['user_id']
            params.update(user_groups)
        return params

    def _get_security_groups(self, element):
        """
        Parse security groups from the provided element and return a
        list of security groups with the id and name key/value pairs.

        :rtype: ``list`` of ``dict``
        """
        groups = []
        for item in findall(element=element, xpath='groupSet/item', namespace=NAMESPACE):
            groups.append({'group_id': findtext(element=item, xpath='groupId', namespace=NAMESPACE), 'group_name': findtext(element=item, xpath='groupName', namespace=NAMESPACE)})
        return groups

    def _build_filters(self, filters):
        """
        Return a dictionary with filter query parameters which are used when
        listing networks, security groups, etc.

        :param      filters: Dict of filter names and filter values
        :type       filters: ``dict``

        :rtype:     ``dict``
        """
        filter_entries = {}
        for filter_idx, filter_data in enumerate(filters.items()):
            filter_idx += 1
            filter_name, filter_values = filter_data
            filter_key = 'Filter.%s.Name' % filter_idx
            filter_entries[filter_key] = filter_name
            if isinstance(filter_values, list):
                for value_idx, value in enumerate(filter_values):
                    value_idx += 1
                    value_key = 'Filter.{}.Value.{}'.format(filter_idx, value_idx)
                    filter_entries[value_key] = value
            else:
                value_key = 'Filter.%s.Value.1' % filter_idx
                filter_entries[value_key] = filter_values
        return filter_entries