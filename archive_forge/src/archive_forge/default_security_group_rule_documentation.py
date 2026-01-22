import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
from openstackclient.network import utils as network_utils
Transform the SDK DefaultSecurityGroupRule object to a dict

        The SDK object gets in the way of reformatting columns...
        Create port_range column from port_range_min and port_range_max
        