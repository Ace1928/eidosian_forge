import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
def validate_ports_match(internal_ports, external_ports):
    internal_ports_diff = validate_ports_diff(internal_ports)
    external_ports_diff = validate_ports_diff(external_ports)
    if internal_ports_diff != 0 and internal_ports_diff != external_ports_diff:
        msg = _('The relation between internal and external ports does not match the pattern 1:N and N:N')
        raise exceptions.CommandError(msg)