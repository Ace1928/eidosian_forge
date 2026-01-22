import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import cliutils
class ShareInstanceDelete(command.Command):
    """Forces the deletion of the share instance."""
    _description = _('Forces the deletion of a share instance')

    def get_parser(self, prog_name):
        parser = super(ShareInstanceDelete, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', nargs='+', help=_('ID of the share instance to delete.'))
        parser.add_argument('--wait', action='store_true', default=False, help=_('Wait for share instance deletion.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        number_of_deletion_failures = 0
        for instance in parsed_args.instance:
            try:
                share_instance = apiutils.find_resource(share_client.share_instances, instance)
                share_client.share_instances.force_delete(share_instance)
                if parsed_args.wait:
                    if not osc_utils.wait_for_delete(manager=share_client.share_instances, res_id=share_instance.id):
                        number_of_deletion_failures += 1
            except Exception as e:
                number_of_deletion_failures += 1
                LOG.error(_("Failed to delete a share instance with ID '%(instance)s': %(e)s"), {'instance': instance, 'e': e})
        if number_of_deletion_failures > 0:
            msg = _('%(number_of_deletion_failures)s of %(total_of_instances)s instances failed to delete.') % {'number_of_deletion_failures': number_of_deletion_failures, 'total_of_instances': len(parsed_args.instance)}
            raise exceptions.CommandError(msg)