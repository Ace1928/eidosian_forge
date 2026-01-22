from osc_lib.command import command
from mistralclient.commands.v2 import base
from mistralclient import utils
def _get_format_function(self):
    return EventTriggerFormatter.format_list