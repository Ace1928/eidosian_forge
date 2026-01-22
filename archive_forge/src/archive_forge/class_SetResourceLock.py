import logging
from openstackclient.identity import common as identity_common
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import constants
class SetResourceLock(command.Command):
    """Set resource lock properties."""
    _description = _('Update resource lock properties')

    def get_parser(self, prog_name):
        parser = super(SetResourceLock, self).get_parser(prog_name)
        parser.add_argument('lock', metavar='<lock>', help='ID of lock to update.')
        parser.add_argument('--resource-action', '--resource_action', metavar='<resource_action>', help='Resource action to set in the resource lock')
        parser.add_argument('--lock-reason', '--lock_reason', '--reason', dest='lock_reason', help='Reason for the resource lock')
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        update_kwargs = {}
        if parsed_args.resource_action is not None:
            update_kwargs['resource_action'] = parsed_args.resource_action
        if parsed_args.lock_reason is not None:
            update_kwargs['lock_reason'] = parsed_args.lock_reason
        if update_kwargs:
            share_client.resource_locks.update(parsed_args.lock, **update_kwargs)