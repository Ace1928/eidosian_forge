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
class ListServer(command.Lister):
    _description = _('List servers')

    def get_parser(self, prog_name):
        parser = super(ListServer, self).get_parser(prog_name)
        parser.add_argument('--reservation-id', metavar='<reservation-id>', help=_('Only return instances that match the reservation'))
        parser.add_argument('--ip', metavar='<ip-address-regex>', help=_('Regular expression to match IP addresses'))
        parser.add_argument('--ip6', metavar='<ip-address-regex>', help=_('Regular expression to match IPv6 addresses. Note that this option only applies for non-admin users when using ``--os-compute-api-version`` 2.5 or greater.'))
        parser.add_argument('--name', metavar='<name-regex>', help=_('Regular expression to match names'))
        parser.add_argument('--instance-name', metavar='<server-name>', help=_('Regular expression to match instance name (admin only)'))
        parser.add_argument('--status', metavar='<status>', choices=('ACTIVE', 'BUILD', 'DELETED', 'ERROR', 'HARD_REBOOT', 'MIGRATING', 'PASSWORD', 'PAUSED', 'REBOOT', 'REBUILD', 'RESCUE', 'RESIZE', 'REVERT_RESIZE', 'SHELVED', 'SHELVED_OFFLOADED', 'SHUTOFF', 'SOFT_DELETED', 'SUSPENDED', 'VERIFY_RESIZE'), help=_('Search by server status'))
        parser.add_argument('--flavor', metavar='<flavor>', help=_('Search by flavor (name or ID)'))
        parser.add_argument('--image', metavar='<image>', help=_('Search by image (name or ID)'))
        parser.add_argument('--host', metavar='<hostname>', help=_('Search by hostname'))
        parser.add_argument('--all-projects', action='store_true', default=boolenv('ALL_PROJECTS'), help=_('Include all projects (admin only) (can be specified using the ALL_PROJECTS envvar)'))
        parser.add_argument('--project', metavar='<project>', help=_('Search by project (admin only) (name or ID)'))
        identity_common.add_project_domain_option_to_parser(parser)
        parser.add_argument('--user', metavar='<user>', help=_('Search by user (name or ID) (admin only before microversion 2.83)'))
        identity_common.add_user_domain_option_to_parser(parser)
        parser.add_argument('--deleted', action='store_true', default=False, help=_('Only display deleted servers (admin only)'))
        parser.add_argument('--availability-zone', default=None, help=_('Search by availability zone (admin only before microversion 2.83)'))
        parser.add_argument('--key-name', help=_('Search by keypair name (admin only before microversion 2.83)'))
        config_drive_group = parser.add_mutually_exclusive_group()
        config_drive_group.add_argument('--config-drive', action='store_true', dest='has_config_drive', default=None, help=_('Only display servers with a config drive attached (admin only before microversion 2.83)'))
        config_drive_group.add_argument('--no-config-drive', action='store_false', dest='has_config_drive', help=_('Only display servers without a config drive attached (admin only before microversion 2.83)'))
        parser.add_argument('--progress', action=PercentAction, default=None, help=_('Search by progress value (%%) (admin only before microversion 2.83)'))
        parser.add_argument('--vm-state', metavar='<state>', choices=('active', 'building', 'deleted', 'error', 'paused', 'stopped', 'suspended', 'rescued', 'resized', 'shelved', 'shelved_offloaded', 'soft-delete'), help=_('Search by vm_state value (admin only before microversion 2.83)'))
        parser.add_argument('--task-state', metavar='<state>', choices=('block_device_mapping', 'deleting', 'image_backup', 'image_pending_upload', 'image_snapshot', 'image_snapshot_pending', 'image_uploading', 'migrating', 'networking', 'pausing', 'powering-off', 'powering-on', 'rebooting', 'reboot_pending', 'reboot_started', 'reboot_pending_hard', 'reboot_started_hard', 'rebooting_hard', 'rebuilding', 'rebuild_block_device_mapping', 'rebuild_spawning', 'rescuing', 'resize_confirming', 'resize_finish', 'resize_migrated', 'resize_migrating', 'resize_prep', 'resize_reverting', 'restoring', 'resuming', 'scheduling', 'shelving', 'shelving_image_pending_upload', 'shelving_image_uploading', 'shelving_offloading', 'soft-deleting', 'spawning', 'suspending', 'updating_password', 'unpausing', 'unrescuing', 'unshelving'), help=_('Search by task_state value (admin only before microversion 2.83)'))
        parser.add_argument('--power-state', metavar='<state>', choices=('pending', 'running', 'paused', 'shutdown', 'crashed', 'suspended'), help=_('Search by power_state value (admin only before microversion 2.83)'))
        parser.add_argument('--long', action='store_true', default=False, help=_('List additional fields in output'))
        name_lookup_group = parser.add_mutually_exclusive_group()
        name_lookup_group.add_argument('-n', '--no-name-lookup', action='store_true', default=False, help=_('Skip flavor and image name lookup. Mutually exclusive with "--name-lookup-one-by-one" option.'))
        name_lookup_group.add_argument('--name-lookup-one-by-one', action='store_true', default=False, help=_('When looking up flavor and image names, look them upone by one as needed instead of all together (default). Mutually exclusive with "--no-name-lookup|-n" option.'))
        pagination.add_marker_pagination_option_to_parser(parser)
        parser.add_argument('--changes-before', metavar='<changes-before>', default=None, help=_('List only servers changed before a certain point of time. The provided time should be an ISO 8061 formatted time (e.g., 2016-03-05T06:27:59Z). (supported by --os-compute-api-version 2.66 or above)'))
        parser.add_argument('--changes-since', metavar='<changes-since>', default=None, help=_('List only servers changed after a certain point of time. The provided time should be an ISO 8061 formatted time (e.g., 2016-03-04T06:27:59Z).'))
        lock_group = parser.add_mutually_exclusive_group()
        lock_group.add_argument('--locked', action='store_true', default=False, help=_('Only display locked servers (supported by --os-compute-api-version 2.73 or above)'))
        lock_group.add_argument('--unlocked', action='store_true', default=False, help=_('Only display unlocked servers (supported by --os-compute-api-version 2.73 or above)'))
        parser.add_argument('--tags', metavar='<tag>', action='append', default=[], dest='tags', help=_('Only list servers with the specified tag. Specify multiple times to filter on multiple tags. (supported by --os-compute-api-version 2.26 or above)'))
        parser.add_argument('--not-tags', metavar='<tag>', action='append', default=[], dest='not_tags', help=_('Only list servers without the specified tag. Specify multiple times to filter on multiple tags. (supported by --os-compute-api-version 2.26 or above)'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        identity_client = self.app.client_manager.identity
        image_client = self.app.client_manager.image
        project_id = None
        if parsed_args.project:
            project_id = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
            parsed_args.all_projects = True
        user_id = None
        if parsed_args.user:
            user_id = identity_common.find_user(identity_client, parsed_args.user, parsed_args.user_domain).id
        flavor_id = None
        if parsed_args.flavor:
            flavor = compute_client.find_flavor(parsed_args.flavor, ignore_missing=False)
            flavor_id = flavor.id
        image_id = None
        if parsed_args.image:
            image_id = image_client.find_image(parsed_args.image, ignore_missing=False).id
        search_opts = {'reservation_id': parsed_args.reservation_id, 'ip': parsed_args.ip, 'ip6': parsed_args.ip6, 'name': parsed_args.name, 'status': parsed_args.status, 'flavor': flavor_id, 'image': image_id, 'host': parsed_args.host, 'project_id': project_id, 'all_projects': parsed_args.all_projects, 'user_id': user_id, 'deleted': parsed_args.deleted, 'changes-before': parsed_args.changes_before, 'changes-since': parsed_args.changes_since}
        if parsed_args.instance_name is not None:
            search_opts['instance_name'] = parsed_args.instance_name
        if parsed_args.availability_zone:
            search_opts['availability_zone'] = parsed_args.availability_zone
        if parsed_args.key_name:
            search_opts['key_name'] = parsed_args.key_name
        if parsed_args.has_config_drive is not None:
            search_opts['config_drive'] = parsed_args.has_config_drive
        if parsed_args.progress is not None:
            search_opts['progress'] = str(parsed_args.progress)
        if parsed_args.vm_state:
            search_opts['vm_state'] = parsed_args.vm_state
        if parsed_args.task_state:
            search_opts['task_state'] = parsed_args.task_state
        if parsed_args.power_state:
            power_state = {'pending': 0, 'running': 1, 'paused': 3, 'shutdown': 4, 'crashed': 6, 'suspended': 7}[parsed_args.power_state]
            search_opts['power_state'] = power_state
        if parsed_args.tags:
            if not sdk_utils.supports_microversion(compute_client, '2.26'):
                msg = _('--os-compute-api-version 2.26 or greater is required to support the --tag option')
                raise exceptions.CommandError(msg)
            search_opts['tags'] = ','.join(parsed_args.tags)
        if parsed_args.not_tags:
            if not sdk_utils.supports_microversion(compute_client, '2.26'):
                msg = _('--os-compute-api-version 2.26 or greater is required to support the --not-tag option')
                raise exceptions.CommandError(msg)
            search_opts['not-tags'] = ','.join(parsed_args.not_tags)
        if parsed_args.locked:
            if not sdk_utils.supports_microversion(compute_client, '2.73'):
                msg = _('--os-compute-api-version 2.73 or greater is required to support the --locked option')
                raise exceptions.CommandError(msg)
            search_opts['locked'] = True
        elif parsed_args.unlocked:
            if not sdk_utils.supports_microversion(compute_client, '2.73'):
                msg = _('--os-compute-api-version 2.73 or greater is required to support the --unlocked option')
                raise exceptions.CommandError(msg)
            search_opts['locked'] = False
        if parsed_args.limit is not None:
            search_opts['limit'] = parsed_args.limit
            search_opts['paginated'] = False
        LOG.debug('search options: %s', search_opts)
        if search_opts['changes-before']:
            if not sdk_utils.supports_microversion(compute_client, '2.66'):
                msg = _('--os-compute-api-version 2.66 or later is required')
                raise exceptions.CommandError(msg)
            try:
                iso8601.parse_date(search_opts['changes-before'])
            except (TypeError, iso8601.ParseError):
                raise exceptions.CommandError(_('Invalid changes-before value: %s') % search_opts['changes-before'])
        if search_opts['changes-since']:
            try:
                iso8601.parse_date(search_opts['changes-since'])
            except (TypeError, iso8601.ParseError):
                msg = _('Invalid changes-since value: %s')
                raise exceptions.CommandError(msg % search_opts['changes-since'])
        columns = ('id', 'name', 'status')
        column_headers = ('ID', 'Name', 'Status')
        if parsed_args.long:
            columns += ('task_state', 'power_state')
            column_headers += ('Task State', 'Power State')
        columns += ('addresses',)
        column_headers += ('Networks',)
        if parsed_args.long:
            columns += ('image_name', 'image_id')
            column_headers += ('Image Name', 'Image ID')
        else:
            if parsed_args.no_name_lookup:
                columns += ('image_id',)
            else:
                columns += ('image_name',)
            column_headers += ('Image',)
        if sdk_utils.supports_microversion(compute_client, '2.47'):
            columns += ('flavor_name',)
            column_headers += ('Flavor',)
        elif parsed_args.long:
            columns += ('flavor_name', 'flavor_id')
            column_headers += ('Flavor Name', 'Flavor ID')
        else:
            if parsed_args.no_name_lookup:
                columns += ('flavor_id',)
            else:
                columns += ('flavor_name',)
            column_headers += ('Flavor',)
        if parsed_args.long:
            columns += ('availability_zone', 'hypervisor_hostname', 'metadata')
            column_headers += ('Availability Zone', 'Host', 'Properties')
        if parsed_args.columns:
            for c in parsed_args.columns:
                if c in ('Project ID', 'project_id'):
                    columns += ('project_id',)
                    column_headers += ('Project ID',)
                if c in ('User ID', 'user_id'):
                    columns += ('user_id',)
                    column_headers += ('User ID',)
                if c in ('Created At', 'created_at'):
                    columns += ('created_at',)
                    column_headers += ('Created At',)
                if c in ('Security Groups', 'security_groups'):
                    columns += ('security_groups_name',)
                    column_headers += ('Security Groups',)
                if c in ('Task State', 'task_state'):
                    columns += ('task_state',)
                    column_headers += ('Task State',)
                if c in ('Power State', 'power_state'):
                    columns += ('power_state',)
                    column_headers += ('Power State',)
                if c in ('Image ID', 'image_id'):
                    columns += ('Image ID',)
                    column_headers += ('Image ID',)
                if c in ('Flavor ID', 'flavor_id'):
                    columns += ('flavor_id',)
                    column_headers += ('Flavor ID',)
                if c in ('Availability Zone', 'availability_zone'):
                    columns += ('availability_zone',)
                    column_headers += ('Availability Zone',)
                if c in ('Host', 'host'):
                    columns += ('hypervisor_hostname',)
                    column_headers += ('Host',)
                if c in ('Properties', 'properties'):
                    columns += ('Metadata',)
                    column_headers += ('Properties',)
            column_headers = tuple(dict.fromkeys(column_headers))
            columns = tuple(dict.fromkeys(columns))
        if parsed_args.marker is not None:
            if parsed_args.deleted:
                marker_id = parsed_args.marker
            else:
                marker_id = compute_client.find_server(parsed_args.marker, ignore_missing=False).id
            search_opts['marker'] = marker_id
        data = list(compute_client.servers(**search_opts))
        images = {}
        flavors = {}
        if data and (not parsed_args.no_name_lookup):
            image_ids = {s.image['id'] for s in data if getattr(s, 'image', None) and s.image.get('id')}
            if parsed_args.name_lookup_one_by_one or image_id:
                for image_id in image_ids:
                    try:
                        images[image_id] = image_client.get_image(image_id)
                    except Exception:
                        pass
            else:
                try:
                    images_list = image_client.images(id=f'in:{','.join(image_ids)}')
                    for i in images_list:
                        images[i.id] = i
                except Exception:
                    pass
            if parsed_args.name_lookup_one_by_one or flavor_id:
                for f_id in set((s.flavor['id'] for s in data if s.flavor and s.flavor.get('id'))):
                    try:
                        flavors[f_id] = compute_client.find_flavor(f_id, ignore_missing=False)
                    except Exception:
                        pass
            else:
                try:
                    flavors_list = compute_client.flavors(is_public=None)
                    for i in flavors_list:
                        flavors[i.id] = i
                except Exception:
                    pass
        for s in data:
            if sdk_utils.supports_microversion(compute_client, '2.69'):
                if not hasattr(s, 'image') or not hasattr(s, 'flavor'):
                    continue
            if 'id' in s.image and s.image.id is not None:
                image = images.get(s.image['id'])
                if image:
                    s.image_name = image.name
                s.image_id = s.image['id']
            else:
                s.image_name = IMAGE_STRING_FOR_BFV
                s.image_id = IMAGE_STRING_FOR_BFV
            if not sdk_utils.supports_microversion(compute_client, '2.47'):
                flavor = flavors.get(s.flavor['id'])
                if flavor:
                    s.flavor_name = flavor.name
                s.flavor_id = s.flavor['id']
            else:
                s.flavor_name = s.flavor['original_name']
        for s in data:
            if hasattr(s, 'security_groups') and s.security_groups is not None:
                s.security_groups_name = [x['name'] for x in s.security_groups]
            else:
                s.security_groups_name = []
        if sdk_utils.supports_microversion(compute_client, '2.16') and parsed_args.long:
            if any([s.host_status is not None for s in data]):
                columns += ('Host Status',)
                column_headers += ('Host Status',)
        table = (column_headers, (utils.get_item_properties(s, columns, mixed_case_fields=('task_state', 'power_state', 'availability_zone', 'host'), formatters={'power_state': PowerStateColumn, 'addresses': AddressesColumn, 'metadata': format_columns.DictColumn, 'security_groups_name': format_columns.ListColumn, 'hypervisor_hostname': HostColumn}) for s in data))
        return table