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
class SetListener(command.Command):
    """Update a listener"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('listener', metavar='<listener>', help='Listener to modify (name or ID).')
        parser.add_argument('--name', metavar='<name>', help='Set the listener name.')
        parser.add_argument('--description', metavar='<description>', help='Set the description of this listener.')
        parser.add_argument('--connection-limit', metavar='<limit>', help='The maximum number of connections permitted for this listener. Default value is -1 which represents infinite connections.')
        parser.add_argument('--default-pool', metavar='<pool>', help='The ID of the pool used by the listener if no L7 policies match.')
        parser.add_argument('--default-tls-container-ref', metavar='<container-ref>', help='The URI to the key manager service secrets container containing the certificate and key for TERMINATED_TLS listeners.')
        parser.add_argument('--sni-container-refs', metavar='<container-ref>', nargs='*', help='A list of URIs to the key manager service secrets containers containing the certificates and keys for TERMINATED_TLS the listener using Server Name Indication.')
        parser.add_argument('--insert-headers', metavar='<header=value>', help='A dictionary of optional headers to insert into the request before it is sent to the backend member.')
        parser.add_argument('--timeout-client-data', type=int, metavar='<timeout>', help='Frontend client inactivity timeout in milliseconds. Default: 50000.')
        parser.add_argument('--timeout-member-connect', type=int, metavar='<timeout>', help='Backend member connection timeout in milliseconds. Default: 5000.')
        parser.add_argument('--timeout-member-data', type=int, metavar='<timeout>', help='Backend member inactivity timeout in milliseconds. Default: 50000.')
        parser.add_argument('--timeout-tcp-inspect', type=int, metavar='<timeout>', help='Time, in milliseconds, to wait for additional TCP packets for content inspection. Default: 0.')
        admin_group = parser.add_mutually_exclusive_group()
        admin_group.add_argument('--enable', action='store_true', default=None, help='Enable listener.')
        admin_group.add_argument('--disable', action='store_true', default=None, help='Disable listener.')
        parser.add_argument('--client-ca-tls-container-ref', metavar='<container_ref>', help='The URI to the key manager service secrets container containing the CA certificate for TERMINATED_TLS listeners.')
        parser.add_argument('--client-authentication', metavar='{' + ','.join(CLIENT_AUTH_CHOICES) + '}', choices=CLIENT_AUTH_CHOICES, type=lambda s: s.upper(), help='The TLS client authentication verify options for TERMINATED_TLS listeners.')
        parser.add_argument('--client-crl-container-ref', metavar='<client_crl_container_ref>', help='The URI to the key manager service secrets container containting the CA revocation list file for TERMINATED_TLS listeners.')
        parser.add_argument('--allowed-cidr', dest='allowed_cidrs', metavar='<allowed_cidr>', nargs='?', action='append', help='CIDR to allow access to the listener (can be set multiple times).')
        parser.add_argument('--wait', action='store_true', help='Wait for action to complete.')
        parser.add_argument('--tls-ciphers', metavar='<tls_ciphers>', help='Set the TLS ciphers to be used by the listener in OpenSSL format.')
        parser.add_argument('--tls-version', dest='tls_versions', metavar='<tls_versions>', nargs='?', action='append', help='Set the TLS protocol version to be used by the listener (can be set multiple times).')
        parser.add_argument('--alpn-protocol', dest='alpn_protocols', metavar='<alpn_protocols>', nargs='?', action='append', help='Set the ALPN protocol to be used by the listener (can be set multiple times).')
        parser.add_argument('--hsts-max-age', dest='hsts_max_age', metavar='<hsts_max_age>', type=int, default=argparse.SUPPRESS, help='The value of the max_age directive for the Strict-Transport-Security HTTP response header. Setting this enables HTTP Strict Transport Security (HSTS) for the TLS-terminated listener.')
        parser.add_argument('--hsts-include-subdomains', action='store_true', default=argparse.SUPPRESS, dest='hsts_include_subdomains', help='Defines whether the includeSubDomains directive should be added to the Strict-Transport-Security HTTP response header.')
        parser.add_argument('--hsts-preload', action='store_true', default=argparse.SUPPRESS, dest='hsts_preload', help='Defines whether the preload directive should be added to the Strict-Transport-Security HTTP response header.')
        _tag.add_tag_option_to_parser_for_set(parser, 'listener')
        return parser

    def take_action(self, parsed_args):
        attrs = v2_utils.get_listener_attrs(self.app.client_manager, parsed_args)
        listener_id = attrs.pop('listener_id')
        v2_utils.set_tags_for_set(self.app.client_manager.load_balancer.listener_show, listener_id, attrs, clear_tags=parsed_args.no_tag)
        body = {'listener': attrs}
        self.app.client_manager.load_balancer.listener_set(listener_id, json=body)
        if parsed_args.wait:
            v2_utils.wait_for_active(status_f=self.app.client_manager.load_balancer.listener_show, res_id=listener_id)