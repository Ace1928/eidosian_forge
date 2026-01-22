import logging
from os_ken.services.protocols.bgp.operator.command import Command
from os_ken.services.protocols.bgp.operator.command import CommandsResponse
from os_ken.services.protocols.bgp.operator.command import STATUS_OK
from os_ken.services.protocols.bgp.operator.command import STATUS_ERROR
from os_ken.services.protocols.bgp.operator.commands.responses import \
class Off(Command):
    command = 'off'
    help_msg = 'turn-off the logging'

    def action(self, params):
        logging.getLogger('bgpspeaker').removeHandler(self.api.log_handler)
        return CommandsResponse(STATUS_OK, True)