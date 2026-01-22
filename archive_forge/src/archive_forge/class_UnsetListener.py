import argparse
from cliff import lister
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
from octaviaclient.osc.v2 import validate
class UnsetListener(command.Command):
    """Clear listener settings"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('listener', metavar='<listener>', help='Listener to modify (name or ID).')
        parser.add_argument('--name', action='store_true', help='Clear the listener name.')
        parser.add_argument('--description', action='store_true', help='Clear the description of this listener.')
        parser.add_argument('--connection-limit', action='store_true', help='Reset the connection limit to the API default.')
        parser.add_argument('--default-pool', dest='default_pool_id', action='store_true', help='Clear the default pool from the listener.')
        parser.add_argument('--default-tls-container-ref', action='store_true', help='Remove the default TLS container reference from the listener.')
        parser.add_argument('--sni-container-refs', action='store_true', help='Remove the TLS SNI container references from the listener.')
        parser.add_argument('--insert-headers', action='store_true', help='Clear the insert headers from the listener.')
        parser.add_argument('--timeout-client-data', action='store_true', help='Reset the client data timeout to the API default.')
        parser.add_argument('--timeout-member-connect', action='store_true', help='Reset the member connect timeout to the API default.')
        parser.add_argument('--timeout-member-data', action='store_true', help='Reset the member data timeout to the API default.')
        parser.add_argument('--timeout-tcp-inspect', action='store_true', help='Reset the TCP inspection timeout to the API default.')
        parser.add_argument('--client-ca-tls-container-ref', action='store_true', help='Clear the client CA TLS container reference from the listener.')
        parser.add_argument('--client-authentication', action='store_true', help='Reset the client authentication setting to the API default.')
        parser.add_argument('--client-crl-container-ref', action='store_true', help='Clear the client CRL container reference from the listener.')
        parser.add_argument('--allowed-cidrs', action='store_true', help='Clear all allowed CIDRs from the listener.')
        parser.add_argument('--tls-versions', action='store_true', help='Clear all TLS versions from the listener.')
        parser.add_argument('--tls-ciphers', action='store_true', help='Clear all TLS ciphers from the listener.')
        parser.add_argument('--wait', action='store_true', help='Wait for action to complete.')
        parser.add_argument('--alpn-protocols', action='store_true', help='Clear all ALPN protocols from the listener.')
        parser.add_argument('--hsts-max-age', dest='hsts_max_age', action='store_true', help='Disables HTTP Strict Transport Security (HSTS) for the TLS-terminated listener.')
        parser.add_argument('--hsts-include-subdomains', action='store_true', dest='hsts_include_subdomains', help='Removes the includeSubDomains directive from the Strict-Transport-Security HTTP response header.')
        parser.add_argument('--hsts-preload', action='store_true', dest='hsts_preload', help='Removes the preload directive from the Strict-Transport-Security HTTP response header.')
        _tag.add_tag_option_to_parser_for_unset(parser, 'listener')
        return parser

    def take_action(self, parsed_args):
        unset_args = v2_utils.get_unsets(parsed_args)
        if not unset_args and (not parsed_args.all_tag):
            return
        listener_id = v2_utils.get_resource_id(self.app.client_manager.load_balancer.listener_list, 'listeners', parsed_args.listener)
        v2_utils.set_tags_for_unset(self.app.client_manager.load_balancer.listener_show, listener_id, unset_args, clear_tags=parsed_args.all_tag)
        body = {'listener': unset_args}
        self.app.client_manager.load_balancer.listener_set(listener_id, json=body)
        if parsed_args.wait:
            v2_utils.wait_for_active(status_f=self.app.client_manager.load_balancer.listener_show, res_id=listener_id)