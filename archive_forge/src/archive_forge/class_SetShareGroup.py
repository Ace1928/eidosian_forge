import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
class SetShareGroup(command.Command):
    """Set share group."""
    _description = _('Explicitly set share group status')

    def get_parser(self, prog_name):
        parser = super(SetShareGroup, self).get_parser(prog_name)
        parser.add_argument('share_group', metavar='<share-group>', help=_('Name or ID of the share group to update.'))
        parser.add_argument('--name', metavar='<name>', default=None, help=_('New name for the share group. (Default=None)'))
        parser.add_argument('--description', metavar='<description>', default=None, help=_('Share group description. (Default=None)'))
        parser.add_argument('--status', metavar='<status>', default=None, help=_('Explicitly update the status of a share group (Admin  only). Examples include: available, error, creating, deleting, error_deleting.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        result = 0
        share_group = osc_utils.find_resource(share_client.share_groups, parsed_args.share_group)
        kwargs = {}
        if parsed_args.name is not None:
            kwargs['name'] = parsed_args.name
        if parsed_args.description is not None:
            kwargs['description'] = parsed_args.description
        if kwargs:
            try:
                share_client.share_groups.update(share_group.id, **kwargs)
            except Exception as e:
                LOG.error(_('Failed to update share group name or description: %s'), e)
                result += 1
        if parsed_args.status:
            try:
                share_group.reset_state(parsed_args.status)
            except Exception as e:
                LOG.error(_('Failed to set status for the share group: %s'), e)
                result += 1
        if result > 0:
            raise exceptions.CommandError(_('One or more of the set operations failed'))