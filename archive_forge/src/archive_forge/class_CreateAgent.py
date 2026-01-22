import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class CreateAgent(command.ShowOne):
    """Create compute agent.

    The compute agent functionality is hypervisor specific and is only
    supported by the XenAPI hypervisor driver. It was removed from nova in the
    23.0.0 (Wallaby) release.
    """

    def get_parser(self, prog_name):
        parser = super(CreateAgent, self).get_parser(prog_name)
        parser.add_argument('os', metavar='<os>', help=_('Type of OS'))
        parser.add_argument('architecture', metavar='<architecture>', help=_('Type of architecture'))
        parser.add_argument('version', metavar='<version>', help=_('Version'))
        parser.add_argument('url', metavar='<url>', help=_('URL'))
        parser.add_argument('md5hash', metavar='<md5hash>', help=_('MD5 hash'))
        parser.add_argument('hypervisor', metavar='<hypervisor>', default='xen', help=_('Type of hypervisor'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.compute
        args = (parsed_args.os, parsed_args.architecture, parsed_args.version, parsed_args.url, parsed_args.md5hash, parsed_args.hypervisor)
        agent = compute_client.agents.create(*args)._info.copy()
        return zip(*sorted(agent.items()))