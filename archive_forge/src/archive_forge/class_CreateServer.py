import argparse
import getpass
import io
import json
import logging
import os
from cliff import columns as cliff_columns
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common as network_common
class CreateServer(command.ShowOne):
    _description = _('Create a new server')

    def get_parser(self, prog_name):
        parser = super(CreateServer, self).get_parser(prog_name)
        parser.add_argument('server_name', metavar='<server-name>', help=_('New server name'))
        parser.add_argument('--flavor', metavar='<flavor>', required=True, help=_('Create server with this flavor (name or ID)'))
        disk_group = parser.add_mutually_exclusive_group()
        disk_group.add_argument('--image', metavar='<image>', help=_('Create server boot disk from this image (name or ID)'))
        disk_group.add_argument('--image-property', metavar='<key=value>', action=parseractions.KeyValueAction, dest='image_properties', help=_('Create server using the image that matches the specified property. Property must match exactly one property.'))
        disk_group.add_argument('--volume', metavar='<volume>', help=_('Create server using this volume as the boot disk (name or ID)\nThis option automatically creates a block device mapping with a boot index of 0. On many hypervisors (libvirt/kvm for example) this will be device vda. Do not create a duplicate mapping using --block-device-mapping for this volume.'))
        disk_group.add_argument('--snapshot', metavar='<snapshot>', help=_('Create server using this snapshot as the boot disk (name or ID)\nThis option automatically creates a block device mapping with a boot index of 0. On many hypervisors (libvirt/kvm for example) this will be device vda. Do not create a duplicate mapping using --block-device-mapping for this volume.'))
        parser.add_argument('--boot-from-volume', metavar='<volume-size>', type=int, help=_('When used in conjunction with the ``--image`` or ``--image-property`` option, this option automatically creates a block device mapping with a boot index of 0 and tells the compute service to create a volume of the given size (in GB) from the specified image and use it as the root disk of the server. The root volume will not be deleted when the server is deleted. This option is mutually exclusive with the ``--volume`` and ``--snapshot`` options.'))
        parser.add_argument('--block-device-mapping', metavar='<dev-name=mapping>', action=BDMLegacyAction, default=[], help=_('**Deprecated** Create a block device on the server.\nBlock device mapping in the format\n<dev-name>=<id>:<type>:<size(GB)>:<delete-on-terminate>\n<dev-name>: block device name, like: vdb, xvdc (required)\n<id>: Name or ID of the volume, volume snapshot or image (required)\n<type>: volume, snapshot or image; default: volume (optional)\n<size(GB)>: volume size if create from image or snapshot (optional)\n<delete-on-terminate>: true or false; default: false (optional)\nReplaced by --block-device'))
        parser.add_argument('--block-device', metavar='', action=BDMAction, dest='block_devices', default=[], help=_('Create a block device on the server.\nEither a path to a JSON file or a CSV-serialized string describing the block device mapping.\nThe following keys are accepted for both:\nuuid=<uuid>: UUID of the volume, snapshot or ID (required if using source image, snapshot or volume),\nsource_type=<source_type>: source type (one of: image, snapshot, volume, blank),\ndestination_type=<destination_type>: destination type (one of: volume, local) (optional),\ndisk_bus=<disk_bus>: device bus (one of: uml, lxc, virtio, ...) (optional),\ndevice_type=<device_type>: device type (one of: disk, cdrom, etc. (optional),\ndevice_name=<device_name>: name of the device (optional),\nvolume_size=<volume_size>: size of the block device in MiB (for swap) or GiB (for everything else) (optional),\nguest_format=<guest_format>: format of device (optional),\nboot_index=<boot_index>: index of disk used to order boot disk (required for volume-backed instances),\ndelete_on_termination=<true|false>: whether to delete the volume upon deletion of server (optional),\ntag=<tag>: device metadata tag (optional),\nvolume_type=<volume_type>: type of volume to create (name or ID) when source if blank, image or snapshot and dest is volume (optional)'))
        parser.add_argument('--swap', metavar='<swap>', type=int, help='Create and attach a local swap block device of <swap_size> MiB.')
        parser.add_argument('--ephemeral', metavar='<size=size[,format=format]>', action=parseractions.MultiKeyValueAction, dest='ephemerals', default=[], required_keys=['size'], optional_keys=['format'], help='Create and attach a local ephemeral block device of <size> GiB and format it to <format>.')
        parser.add_argument('--network', metavar='<network>', dest='nics', action=NICAction, key='net-id', help=_("Create a NIC on the server and connect it to network. Specify option multiple times to create multiple NICs. This is a wrapper for the '--nic net-id=<network>' parameter that provides simple syntax for the standard use case of connecting a new server to a given network. For more advanced use cases, refer to the '--nic' parameter."))
        parser.add_argument('--port', metavar='<port>', dest='nics', action=NICAction, key='port-id', help=_("Create a NIC on the server and connect it to port. Specify option multiple times to create multiple NICs. This is a wrapper for the '--nic port-id=<port>' parameter that provides simple syntax for the standard use case of connecting a new server to a given port. For more advanced use cases, refer to the '--nic' parameter."))
        parser.add_argument('--no-network', dest='nics', action=NoneNICAction, help=_("Do not attach a network to the server. This is a wrapper for the '--nic none' option that provides a simple syntax for disabling network connectivity for a new server. For more advanced use cases, refer to the '--nic' parameter. (supported by --os-compute-api-version 2.37 or above)"))
        parser.add_argument('--auto-network', dest='nics', action=AutoNICAction, help=_("Automatically allocate a network to the server. This is the default network allocation policy. This is a wrapper for the '--nic auto' option that provides a simple syntax for enabling automatic configuration of network connectivity for a new server. For more advanced use cases, refer to the '--nic' parameter. (supported by --os-compute-api-version 2.37 or above)"))
        parser.add_argument('--nic', metavar='<net-id=net-uuid,port-id=port-uuid,v4-fixed-ip=ip-addr,v6-fixed-ip=ip-addr,tag=tag,auto,none>', dest='nics', action=NICAction, help=_('Create a NIC on the server.\nNIC in the format:\nnet-id=<net-uuid>: attach NIC to network with this UUID,\nport-id=<port-uuid>: attach NIC to port with this UUID,\nv4-fixed-ip=<ip-addr>: IPv4 fixed address for NIC (optional),\nv6-fixed-ip=<ip-addr>: IPv6 fixed address for NIC (optional),\ntag: interface metadata tag (optional) (supported by --os-compute-api-version 2.43 or above),\nnone: (v2.37+) no network is attached,\nauto: (v2.37+) the compute service will automatically allocate a network.\n\nSpecify option multiple times to create multiple NICs.\nSpecifying a --nic of auto or none cannot be used with any other --nic value.\nEither net-id or port-id must be provided, but not both.'))
        parser.add_argument('--password', metavar='<password>', help=_('Set the password to this server. This option requires cloud support.'))
        parser.add_argument('--security-group', metavar='<security-group>', action='append', default=[], help=_('Security group to assign to this server (name or ID) (repeat option to set multiple groups)'))
        parser.add_argument('--key-name', metavar='<key-name>', help=_('Keypair to inject into this server'))
        parser.add_argument('--property', metavar='<key=value>', action=parseractions.KeyValueAction, dest='properties', help=_('Set a property on this server (repeat option to set multiple values)'))
        parser.add_argument('--file', metavar='<dest-filename=source-filename>', action='append', default=[], help=_('File(s) to inject into image before boot (repeat option to set multiple files)(supported by --os-compute-api-version 2.57 or below)'))
        parser.add_argument('--user-data', metavar='<user-data>', help=_('User data file to serve from the metadata server'))
        parser.add_argument('--description', metavar='<description>', help=_('Set description for the server (supported by --os-compute-api-version 2.19 or above)'))
        parser.add_argument('--availability-zone', metavar='<zone-name>', help=_('Select an availability zone for the server. Host and node are optional parameters. Availability zone in the format <zone-name>:<host-name>:<node-name>, <zone-name>::<node-name>, <zone-name>:<host-name> or <zone-name>'))
        parser.add_argument('--host', metavar='<host>', help=_('Requested host to create servers. (admin only) (supported by --os-compute-api-version 2.74 or above)'))
        parser.add_argument('--hypervisor-hostname', metavar='<hypervisor-hostname>', help=_('Requested hypervisor hostname to create servers. (admin only) (supported by --os-compute-api-version 2.74 or above)'))
        parser.add_argument('--server-group', metavar='<server-group>', help=_("Server group to create the server within (this is an alias for '--hint group=<server-group-id>')"))
        parser.add_argument('--hint', metavar='<key=value>', action=parseractions.KeyValueAppendAction, dest='hints', default={}, help=_('Hints for the scheduler'))
        config_drive_group = parser.add_mutually_exclusive_group()
        config_drive_group.add_argument('--use-config-drive', action='store_true', dest='config_drive', help=_('Enable config drive.'))
        config_drive_group.add_argument('--no-config-drive', action='store_false', dest='config_drive', help=_('Disable config drive.'))
        config_drive_group.add_argument('--config-drive', metavar='<config-drive-volume>|True', default=False, help=_("**Deprecated** Use specified volume as the config drive, or 'True' to use an ephemeral drive. Replaced by '--use-config-drive'."))
        parser.add_argument('--min', metavar='<count>', type=int, default=1, help=_('Minimum number of servers to launch (default=1)'))
        parser.add_argument('--max', metavar='<count>', type=int, default=1, help=_('Maximum number of servers to launch (default=1)'))
        parser.add_argument('--tag', metavar='<tag>', action='append', default=[], dest='tags', help=_('Tags for the server. Specify multiple times to add multiple tags. (supported by --os-compute-api-version 2.52 or above)'))
        parser.add_argument('--hostname', metavar='<hostname>', help=_('Hostname configured for the server in the metadata service. If unset, a hostname will be automatically generated from the server name. A utility such as cloud-init is required to propagate the hostname in the metadata service to the guest OS itself. (supported by --os-compute-api-version 2.90 or above)'))
        parser.add_argument('--wait', action='store_true', help=_('Wait for build to complete'))
        parser.add_argument('--trusted-image-cert', metavar='<trusted-cert-id>', action='append', dest='trusted_image_certs', help=_('Trusted image certificate IDs used to validate certificates during the image signature verification process. May be specified multiple times to pass multiple trusted image certificate IDs. (supported by --os-compute-api-version 2.63 or above)'))
        return parser

    def take_action(self, parsed_args):

        def _show_progress(progress):
            if progress:
                self.app.stdout.write('\rProgress: %s' % progress)
                self.app.stdout.flush()
        compute_client = self.app.client_manager.compute
        volume_client = self.app.client_manager.volume
        image_client = self.app.client_manager.image
        image = None
        if parsed_args.image:
            image = image_client.find_image(parsed_args.image, ignore_missing=False)
        if not image and parsed_args.image_properties:

            def emit_duplicated_warning(img):
                img_uuid_list = [str(image.id) for image in img]
                LOG.warning('Multiple matching images: %(img_uuid_list)s\nUsing image: %(chosen_one)s', {'img_uuid_list': img_uuid_list, 'chosen_one': img_uuid_list[0]})

            def _match_image(image_api, wanted_properties):
                image_list = image_api.images()
                images_matched = []
                for img in image_list:
                    img_dict = {}
                    img_dict_items = list(img.items())
                    if img.properties:
                        img_dict_items.extend(list(img.properties.items()))
                    for key, value in img_dict_items:
                        try:
                            set([key, value])
                        except TypeError:
                            if key != 'properties':
                                LOG.debug("Skipped the '%s' attribute. That cannot be compared. (image: %s, value: %s)", key, img.id, value)
                            pass
                        else:
                            img_dict[key] = value
                    if all((k in img_dict and img_dict[k] == v for k, v in wanted_properties.items())):
                        images_matched.append(img)
                return images_matched
            images = _match_image(image_client, parsed_args.image_properties)
            if len(images) > 1:
                emit_duplicated_warning(images, parsed_args.image_properties)
            if images:
                image = images[0]
            else:
                msg = _('No images match the property expected by --image-property')
                raise exceptions.CommandError(msg)
        volume = None
        if parsed_args.volume:
            if parsed_args.boot_from_volume:
                msg = _('--volume is not allowed with --boot-from-volume')
                raise exceptions.CommandError(msg)
            volume = utils.find_resource(volume_client.volumes, parsed_args.volume).id
        snapshot = None
        if parsed_args.snapshot:
            if parsed_args.boot_from_volume:
                msg = _('--snapshot is not allowed with --boot-from-volume')
                raise exceptions.CommandError(msg)
            snapshot = utils.find_resource(volume_client.volume_snapshots, parsed_args.snapshot).id
        flavor = utils.find_resource(compute_client.flavors, parsed_args.flavor)
        if parsed_args.file:
            if compute_client.api_version >= api_versions.APIVersion('2.57'):
                msg = _('Personality files are deprecated and are not supported for --os-compute-api-version greater than 2.56; use user data instead')
                raise exceptions.CommandError(msg)
        files = {}
        for f in parsed_args.file:
            dst, src = f.split('=', 1)
            try:
                files[dst] = io.open(src, 'rb')
            except IOError as e:
                msg = _("Can't open '%(source)s': %(exception)s")
                raise exceptions.CommandError(msg % {'source': src, 'exception': e})
        if parsed_args.min > parsed_args.max:
            msg = _('min instances should be <= max instances')
            raise exceptions.CommandError(msg)
        if parsed_args.min < 1:
            msg = _('min instances should be > 0')
            raise exceptions.CommandError(msg)
        if parsed_args.max < 1:
            msg = _('max instances should be > 0')
            raise exceptions.CommandError(msg)
        userdata = None
        if parsed_args.user_data:
            try:
                userdata = io.open(parsed_args.user_data)
            except IOError as e:
                msg = _("Can't open '%(data)s': %(exception)s")
                raise exceptions.CommandError(msg % {'data': parsed_args.user_data, 'exception': e})
        if parsed_args.description:
            if compute_client.api_version < api_versions.APIVersion('2.19'):
                msg = _('--os-compute-api-version 2.19 or greater is required to support the --description option')
                raise exceptions.CommandError(msg)
        block_device_mapping_v2 = []
        if volume:
            block_device_mapping_v2 = [{'uuid': volume, 'boot_index': 0, 'source_type': 'volume', 'destination_type': 'volume'}]
        elif snapshot:
            block_device_mapping_v2 = [{'uuid': snapshot, 'boot_index': 0, 'source_type': 'snapshot', 'destination_type': 'volume', 'delete_on_termination': False}]
        elif parsed_args.boot_from_volume:
            if not image:
                msg = _('An image (--image or --image-property) is required to support --boot-from-volume option')
                raise exceptions.CommandError(msg)
            block_device_mapping_v2 = [{'uuid': image.id, 'boot_index': 0, 'source_type': 'image', 'destination_type': 'volume', 'volume_size': parsed_args.boot_from_volume}]
            image = None
        if parsed_args.swap:
            block_device_mapping_v2.append({'boot_index': -1, 'source_type': 'blank', 'destination_type': 'local', 'guest_format': 'swap', 'volume_size': parsed_args.swap, 'delete_on_termination': True})
        for mapping in parsed_args.ephemerals:
            block_device_mapping_dict = {'boot_index': -1, 'source_type': 'blank', 'destination_type': 'local', 'delete_on_termination': True, 'volume_size': mapping['size']}
            if 'format' in mapping:
                block_device_mapping_dict['guest_format'] = mapping['format']
            block_device_mapping_v2.append(block_device_mapping_dict)
        for mapping in parsed_args.block_device_mapping:
            if mapping['source_type'] == 'volume':
                volume_id = utils.find_resource(volume_client.volumes, mapping['uuid']).id
                mapping['uuid'] = volume_id
            elif mapping['source_type'] == 'snapshot':
                snapshot_id = utils.find_resource(volume_client.volume_snapshots, mapping['uuid']).id
                mapping['uuid'] = snapshot_id
            elif mapping['source_type'] == 'image':
                image_id = image_client.find_image(mapping['uuid'], ignore_missing=False).id
                mapping['uuid'] = image_id
            block_device_mapping_v2.append(mapping)
        for mapping in parsed_args.block_devices:
            if 'boot_index' in mapping:
                try:
                    mapping['boot_index'] = int(mapping['boot_index'])
                except ValueError:
                    msg = _('The boot_index key of --block-device should be an integer')
                    raise exceptions.CommandError(msg)
            if 'tag' in mapping and compute_client.api_version < api_versions.APIVersion('2.42'):
                msg = _('--os-compute-api-version 2.42 or greater is required to support the tag key of --block-device')
                raise exceptions.CommandError(msg)
            if 'volume_type' in mapping and compute_client.api_version < api_versions.APIVersion('2.67'):
                msg = _('--os-compute-api-version 2.67 or greater is required to support the volume_type key of --block-device')
                raise exceptions.CommandError(msg)
            if 'source_type' in mapping:
                if mapping['source_type'] not in ('volume', 'image', 'snapshot', 'blank'):
                    msg = _('The source_type key of --block-device should be one of: volume, image, snapshot, blank')
                    raise exceptions.CommandError(msg)
            else:
                mapping['source_type'] = 'blank'
            if 'destination_type' in mapping:
                if mapping['destination_type'] not in ('local', 'volume'):
                    msg = _('The destination_type key of --block-device should be one of: local, volume')
                    raise exceptions.CommandError(msg)
            elif mapping['source_type'] in ('blank',):
                mapping['destination_type'] = 'local'
            else:
                mapping['destination_type'] = 'volume'
            if 'delete_on_termination' in mapping:
                try:
                    value = bool_from_str(mapping['delete_on_termination'], strict=True)
                except ValueError:
                    msg = _('The delete_on_termination key of --block-device should be a boolean-like value')
                    raise exceptions.CommandError(msg)
                mapping['delete_on_termination'] = value
            elif mapping['destination_type'] == 'local':
                mapping['delete_on_termination'] = True
            block_device_mapping_v2.append(mapping)
        if not image and (not any([bdm.get('boot_index') == 0 for bdm in block_device_mapping_v2])):
            msg = _('An image (--image, --image-property) or bootable volume (--volume, --snapshot, --block-device) is required')
            raise exceptions.CommandError(msg)
        nics = parsed_args.nics
        if 'auto' in nics or 'none' in nics:
            if len(nics) > 1:
                msg = _('Specifying a --nic of auto or none cannot be used with any other --nic, --network or --port value.')
                raise exceptions.CommandError(msg)
            if compute_client.api_version < api_versions.APIVersion('2.37'):
                msg = _('--os-compute-api-version 2.37 or greater is required to support explicit auto-allocation of a network or to disable network allocation')
                raise exceptions.CommandError(msg)
            nics = nics[0]
        else:
            for nic in nics:
                if 'tag' in nic:
                    if compute_client.api_version < api_versions.APIVersion('2.43'):
                        msg = _('--os-compute-api-version 2.43 or greater is required to support the --nic tag field')
                        raise exceptions.CommandError(msg)
                if self.app.client_manager.is_network_endpoint_enabled():
                    network_client = self.app.client_manager.network
                    if nic['net-id']:
                        net = network_client.find_network(nic['net-id'], ignore_missing=False)
                        nic['net-id'] = net.id
                    if nic['port-id']:
                        port = network_client.find_port(nic['port-id'], ignore_missing=False)
                        nic['port-id'] = port.id
                else:
                    if nic['net-id']:
                        nic['net-id'] = compute_client.api.network_find(nic['net-id'])['id']
                    if nic['port-id']:
                        msg = _("Can't create server with port specified since network endpoint not enabled")
                        raise exceptions.CommandError(msg)
        if not nics:
            if compute_client.api_version >= api_versions.APIVersion('2.37'):
                nics = 'auto'
            else:
                nics = []
        security_group_names = []
        if self.app.client_manager.is_network_endpoint_enabled():
            network_client = self.app.client_manager.network
            for each_sg in parsed_args.security_group:
                sg = network_client.find_security_group(each_sg, ignore_missing=False)
                security_group_names.append(sg.id)
        else:
            for each_sg in parsed_args.security_group:
                sg = compute_client.api.security_group_find(each_sg)
                security_group_names.append(sg['name'])
        hints = {}
        for key, values in parsed_args.hints.items():
            if len(values) == 1:
                hints[key] = values[0]
            else:
                hints[key] = values
        if parsed_args.server_group:
            server_group_obj = utils.find_resource(compute_client.server_groups, parsed_args.server_group)
            hints['group'] = server_group_obj.id
        if isinstance(parsed_args.config_drive, bool):
            config_drive = parsed_args.config_drive or None
        elif str(parsed_args.config_drive).lower() in ('true', '1'):
            config_drive = True
        elif str(parsed_args.config_drive).lower() in ('false', '0', '', 'none'):
            config_drive = None
        else:
            config_drive = parsed_args.config_drive
        boot_args = [parsed_args.server_name, image, flavor]
        boot_kwargs = dict(meta=parsed_args.properties, files=files, reservation_id=None, min_count=parsed_args.min, max_count=parsed_args.max, security_groups=security_group_names, userdata=userdata, key_name=parsed_args.key_name, availability_zone=parsed_args.availability_zone, admin_pass=parsed_args.password, block_device_mapping_v2=block_device_mapping_v2, nics=nics, scheduler_hints=hints, config_drive=config_drive)
        if parsed_args.description:
            boot_kwargs['description'] = parsed_args.description
        if parsed_args.tags:
            if compute_client.api_version < api_versions.APIVersion('2.52'):
                msg = _('--os-compute-api-version 2.52 or greater is required to support the --tag option')
                raise exceptions.CommandError(msg)
            boot_kwargs['tags'] = parsed_args.tags
        if parsed_args.host:
            if compute_client.api_version < api_versions.APIVersion('2.74'):
                msg = _('--os-compute-api-version 2.74 or greater is required to support the --host option')
                raise exceptions.CommandError(msg)
            boot_kwargs['host'] = parsed_args.host
        if parsed_args.hypervisor_hostname:
            if compute_client.api_version < api_versions.APIVersion('2.74'):
                msg = _('--os-compute-api-version 2.74 or greater is required to support the --hypervisor-hostname option')
                raise exceptions.CommandError(msg)
            boot_kwargs['hypervisor_hostname'] = parsed_args.hypervisor_hostname
        if parsed_args.hostname:
            if compute_client.api_version < api_versions.APIVersion('2.90'):
                msg = _('--os-compute-api-version 2.90 or greater is required to support the --hostname option')
                raise exceptions.CommandError(msg)
            boot_kwargs['hostname'] = parsed_args.hostname
        if parsed_args.trusted_image_certs:
            if not (image and (not parsed_args.boot_from_volume)):
                msg = _('--trusted-image-cert option is only supported for servers booted directly from images')
                raise exceptions.CommandError(msg)
            if compute_client.api_version < api_versions.APIVersion('2.63'):
                msg = _('--os-compute-api-version 2.63 or greater is required to support the --trusted-image-cert option')
                raise exceptions.CommandError(msg)
            certs = parsed_args.trusted_image_certs
            boot_kwargs['trusted_image_certificates'] = certs
        LOG.debug('boot_args: %s', boot_args)
        LOG.debug('boot_kwargs: %s', boot_kwargs)
        try:
            server = compute_client.servers.create(*boot_args, **boot_kwargs)
        finally:
            for f in files:
                if hasattr(f, 'close'):
                    f.close()
            if hasattr(userdata, 'close'):
                userdata.close()
        if parsed_args.wait:
            if utils.wait_for_status(compute_client.servers.get, server.id, callback=_show_progress):
                self.app.stdout.write('\n')
            else:
                msg = _('Error creating server: %s') % parsed_args.server_name
                raise exceptions.CommandError(msg)
        details = _prep_server_detail(compute_client, image_client, server)
        return zip(*sorted(details.items()))