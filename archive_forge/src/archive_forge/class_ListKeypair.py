import collections
import io
import logging
import os
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from openstack import utils as sdk_utils
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class ListKeypair(command.Lister):
    _description = _('List key fingerprints')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        user_group = parser.add_mutually_exclusive_group()
        user_group.add_argument('--user', metavar='<user>', help=_('Show keypairs for another user (admin only) (name or ID). Requires ``--os-compute-api-version`` 2.10 or greater.'))
        identity_common.add_user_domain_option_to_parser(parser)
        user_group.add_argument('--project', metavar='<project>', help=_('Show keypairs for all users associated with project (admin only) (name or ID). Requires ``--os-compute-api-version`` 2.10 or greater.'))
        identity_common.add_project_domain_option_to_parser(parser)
        pagination.add_marker_pagination_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        identity_client = self.app.client_manager.identity
        kwargs = {}
        if parsed_args.marker:
            if not sdk_utils.supports_microversion(compute_client, '2.35'):
                msg = _('--os-compute-api-version 2.35 or greater is required to support the --marker option')
                raise exceptions.CommandError(msg)
            kwargs['marker'] = parsed_args.marker
        if parsed_args.limit:
            if not sdk_utils.supports_microversion(compute_client, '2.35'):
                msg = _('--os-compute-api-version 2.35 or greater is required to support the --limit option')
                raise exceptions.CommandError(msg)
            kwargs['limit'] = parsed_args.limit
        if parsed_args.project:
            if not sdk_utils.supports_microversion(compute_client, '2.10'):
                msg = _('--os-compute-api-version 2.10 or greater is required to support the --project option')
                raise exceptions.CommandError(msg)
            if parsed_args.marker:
                msg = _('--project is not compatible with --marker')
            project = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
            users = identity_client.users.list(tenant_id=project)
            data = []
            for user in users:
                kwargs['user_id'] = user.id
                data.extend(compute_client.keypairs(**kwargs))
        elif parsed_args.user:
            if not sdk_utils.supports_microversion(compute_client, '2.10'):
                msg = _('--os-compute-api-version 2.10 or greater is required to support the --user option')
                raise exceptions.CommandError(msg)
            user = identity_common.find_user(identity_client, parsed_args.user, parsed_args.user_domain)
            kwargs['user_id'] = user.id
            data = compute_client.keypairs(**kwargs)
        else:
            data = compute_client.keypairs(**kwargs)
        columns = ('Name', 'Fingerprint')
        if sdk_utils.supports_microversion(compute_client, '2.2'):
            columns += ('Type',)
        return (columns, (utils.get_item_properties(s, columns) for s in data))