from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient.i18n import _
from osc_lib.command import command
def _show_quota(quota):
    utils.print_dict(quota._info)