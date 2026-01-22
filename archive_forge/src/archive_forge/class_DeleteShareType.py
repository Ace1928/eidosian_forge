import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import constants
from manilaclient.osc import utils
class DeleteShareType(command.Command):
    """Delete a share type."""
    _description = _('Delete a share type')

    def get_parser(self, prog_name):
        parser = super(DeleteShareType, self).get_parser(prog_name)
        parser.add_argument('share_types', metavar='<share_types>', nargs='+', help=_('Name or ID of the share type(s) to delete'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        result = 0
        for share_type in parsed_args.share_types:
            try:
                share_type_obj = apiutils.find_resource(share_client.share_types, share_type)
                share_client.share_types.delete(share_type_obj)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete share type with name or ID '%(share_type)s': %(e)s"), {'share_type': share_type, 'e': e})
        if result > 0:
            total = len(parsed_args.share_types)
            msg = _('%(result)s of %(total)s share types failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)