from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
def _show_action(action):
    utils.print_dict(action._info)