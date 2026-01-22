import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class SetAgent(command.Command):
    """Set compute agent properties.

    The compute agent functionality is hypervisor specific and is only
    supported by the XenAPI hypervisor driver. It was removed from nova in the
    23.0.0 (Wallaby) release.
    """

    def get_parser(self, prog_name):
        parser = super(SetAgent, self).get_parser(prog_name)
        parser.add_argument('id', metavar='<id>', help=_('ID of the agent'))
        parser.add_argument('--agent-version', dest='version', metavar='<version>', help=_('Version of the agent'))
        parser.add_argument('--url', metavar='<url>', help=_('URL of the agent'))
        parser.add_argument('--md5hash', metavar='<md5hash>', help=_('MD5 hash of the agent'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.compute
        data = compute_client.agents.list(hypervisor=None)
        agent = {}
        for s in data:
            if s.agent_id == int(parsed_args.id):
                agent['version'] = s.version
                agent['url'] = s.url
                agent['md5hash'] = s.md5hash
        if agent == {}:
            msg = _("No agent with a ID of '%(id)s' exists.")
            raise exceptions.CommandError(msg % parsed_args.id)
        if parsed_args.version:
            agent['version'] = parsed_args.version
        if parsed_args.url:
            agent['url'] = parsed_args.url
        if parsed_args.md5hash:
            agent['md5hash'] = parsed_args.md5hash
        args = (parsed_args.id, agent['version'], agent['url'], agent['md5hash'])
        compute_client.agents.update(*args)