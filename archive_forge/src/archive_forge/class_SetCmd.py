import logging
from os_ken.services.protocols.bgp.operator.command import Command
from os_ken.services.protocols.bgp.operator.command import CommandsResponse
from os_ken.services.protocols.bgp.operator.command import STATUS_OK
from os_ken.services.protocols.bgp.operator.command import STATUS_ERROR
from os_ken.services.protocols.bgp.operator.commands.responses import \
class SetCmd(Command):
    help_msg = 'set runtime settings'
    command = 'set'
    subcommands = {'logging': LoggingCmd}

    def action(self, params):
        return CommandsResponse(STATUS_ERROR, 'Command incomplete')