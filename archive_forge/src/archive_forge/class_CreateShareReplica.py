import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common import cliutils
from manilaclient.osc import utils
class CreateShareReplica(command.ShowOne):
    """Create a share replica."""
    _description = _('Create a replica of the given share')

    def get_parser(self, prog_name):
        parser = super(CreateShareReplica, self).get_parser(prog_name)
        parser.add_argument('share', metavar='<share>', help=_('Name or ID of the share to replicate.'))
        parser.add_argument('--availability-zone', metavar='<availability-zone>', default=None, help=_('Availability zone in which the replica should be created.'))
        parser.add_argument('--wait', action='store_true', default=False, help=_('Wait for replica creation'))
        parser.add_argument('--scheduler-hint', metavar='<key=value>', default={}, action=parseractions.KeyValueAction, help=_('Scheduler hint for the share replica as key=value pairs, Supported key is only_host. Available for microversion >= 2.67.'))
        parser.add_argument('--share-network', metavar='<share-network-name-or-id>', default=None, help=_('Optional network info ID or name. Available for microversion >= 2.72'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share = osc_utils.find_resource(share_client.shares, parsed_args.share)
        scheduler_hints = {}
        if parsed_args.scheduler_hint:
            if share_client.api_version < api_versions.APIVersion('2.67'):
                raise exceptions.CommandError(_("arg '--scheduler_hint' is available only starting with API microversion '2.67'."))
            hints = utils.extract_key_value_options(parsed_args.scheduler_hint)
            if 'only_host' not in hints.keys() or len(hints) > 1:
                raise exceptions.CommandError("The only valid key supported with the --scheduler-hint argument is 'only_host'.")
            scheduler_hints['only_host'] = hints.get('only_host')
        body = {'share': share, 'availability_zone': parsed_args.availability_zone}
        if scheduler_hints:
            body['scheduler_hints'] = scheduler_hints
        share_network_id = None
        if parsed_args.share_network:
            if share_client.api_version < api_versions.APIVersion('2.72'):
                raise exceptions.CommandError("'share-network' option is available only starting with '2.72' API microversion.")
            share_network_id = osc_utils.find_resource(share_client.share_networks, parsed_args.share_network).id
            body['share_network'] = share_network_id
        share_replica = share_client.share_replicas.create(**body)
        if parsed_args.wait:
            if not osc_utils.wait_for_status(status_f=share_client.share_replicas.get, res_id=share_replica.id, success_status=['available']):
                LOG.error(_('ERROR: Share replica is in error state.'))
            share_replica = osc_utils.find_resource(share_client.share_replicas, share_replica.id)
        return self.dict2columns(share_replica._info)