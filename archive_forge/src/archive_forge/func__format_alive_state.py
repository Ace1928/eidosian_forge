from osc_lib.command import command
from osc_lib import utils
from neutronclient._i18n import _
def _format_alive_state(item):
    return ':-)' if item else 'XXX'