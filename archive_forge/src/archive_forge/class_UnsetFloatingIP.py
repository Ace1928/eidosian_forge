from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class UnsetFloatingIP(common.NeutronCommandWithExtraArgs):
    _description = _('Unset floating IP Properties')

    def get_parser(self, prog_name):
        parser = super(UnsetFloatingIP, self).get_parser(prog_name)
        parser.add_argument('floating_ip', metavar='<floating-ip>', help=_('Floating IP to disassociate (IP address or ID)'))
        parser.add_argument('--port', action='store_true', default=False, help=_('Disassociate any port associated with the floating IP'))
        parser.add_argument('--qos-policy', action='store_true', default=False, help=_('Remove the QoS policy attached to the floating IP'))
        _tag.add_tag_option_to_parser_for_unset(parser, _('floating IP'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_ip(parsed_args.floating_ip, ignore_missing=False)
        attrs = {}
        if parsed_args.port:
            attrs['port_id'] = None
        if parsed_args.qos_policy:
            attrs['qos_policy_id'] = None
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        if attrs:
            client.update_ip(obj, **attrs)
        _tag.update_tags_for_unset(client, obj, parsed_args)