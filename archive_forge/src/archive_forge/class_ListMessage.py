import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
class ListMessage(command.Lister):
    """Lists all messages."""
    _description = _('Lists all messages')

    def get_parser(self, prog_name):
        parser = super(ListMessage, self).get_parser(prog_name)
        parser.add_argument('--resource-id', metavar='<resource-id>', default=None, help=_('Filters results by a resource uuid. Default=None.'))
        parser.add_argument('--resource-type', metavar='<resource-type>', default=None, help=_('Filters results by a resource type. Default=None. Example: "openstack message list --resource-type share"'))
        parser.add_argument('--action-id', metavar='<action-id>', default=None, help=_('Filters results by action id. Default=None.'))
        parser.add_argument('--detail-id', metavar='<detail-id>', default=None, help=_('Filters results by detail id. Default=None.'))
        parser.add_argument('--request-id', metavar='<request-id>', default=None, help=_('Filters results by request id. Default=None.'))
        parser.add_argument('--message-level', metavar='<message-level>', default=None, help=_('Filters results by the message level. Default=None. Example: "openstack message list --message-level ERROR".'))
        parser.add_argument('--limit', metavar='<limit>', type=int, default=None, help=_('Maximum number of messages to return. (Default=None)'))
        parser.add_argument('--since', metavar='<since>', default=None, help=_('Return only user messages created since given date. The date format must be conforming to ISO8601. Available only for microversion >= 2.52.'))
        parser.add_argument('--before', metavar='<before>', default=None, help=_('Return only user messages created before given date. The date format must be conforming to ISO8601. Available only for microversion >= 2.52.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        search_opts = {'limit': parsed_args.limit, 'request_id': parsed_args.request_id, 'resource_type': parsed_args.resource_type, 'resource_id': parsed_args.resource_id, 'action_id': parsed_args.action_id, 'detail_id': parsed_args.detail_id, 'message_level': parsed_args.message_level}
        if share_client.api_version < api_versions.APIVersion('2.52'):
            if getattr(parsed_args, 'since') or getattr(parsed_args, 'before'):
                raise exceptions.CommandError(_("Filtering messages by 'since' and 'before' is possible only with Manila API version >=2.52"))
        else:
            search_opts['created_since'] = parsed_args.since
            search_opts['created_before'] = parsed_args.before
        messages = share_client.messages.list(search_opts=search_opts)
        columns = ['ID', 'Resource Type', 'Resource ID', 'Action ID', 'User Message', 'Detail ID', 'Created At']
        return (columns, (oscutils.get_item_properties(m, columns) for m in messages))