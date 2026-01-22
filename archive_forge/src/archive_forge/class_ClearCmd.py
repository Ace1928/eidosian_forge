from os_ken.services.protocols.bgp.operator.command import Command
from os_ken.services.protocols.bgp.operator.command import CommandsResponse
from os_ken.services.protocols.bgp.operator.command import STATUS_OK
from os_ken.services.protocols.bgp.operator.commands.responses import \
class ClearCmd(Command):
    help_msg = 'allows to reset BGP connections'
    command = 'clear'
    subcommands = {'bgp': BGPCmd}