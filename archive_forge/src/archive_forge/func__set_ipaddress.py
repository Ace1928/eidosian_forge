import copy
from oslo_config import cfg
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources import scheduler_hints as sh
def _set_ipaddress(self, networks):
    """Set IP address to self.ipaddress from a list of networks.

        Read the server's IP address from a list of networks provided by Nova.
        """
    for n in sorted(networks, reverse=True):
        if len(networks[n]) > 0:
            self.ipaddress = networks[n][0]
            break