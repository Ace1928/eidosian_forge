from osc_lib.command import command
from mistralclient.commands.v2 import base
from mistralclient import utils
@staticmethod
def _get_json_string_or_dict(string):
    if string:
        return utils.load_json(string)
    else:
        return {}