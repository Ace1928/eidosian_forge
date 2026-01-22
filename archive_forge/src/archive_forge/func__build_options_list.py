import argparse
import itertools
import logging
import sys
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
def _build_options_list(self):
    help_fmt = _('New value for the %s quota')
    rets = [(k, v, help_fmt % v) for k, v in itertools.chain(COMPUTE_QUOTAS.items(), VOLUME_QUOTAS.items())]
    if self.is_docs_build:
        inv_compute = set(NOVA_NETWORK_QUOTAS.values())
        for k, v in NETWORK_QUOTAS.items():
            _help = help_fmt % v
            if v not in inv_compute:
                _help = self.enhance_help_neutron(_help)
            rets.append((k, v, _help))
    elif self.is_neutron:
        rets.extend([(k, v, help_fmt % v) for k, v in NETWORK_QUOTAS.items()])
    elif self.is_nova_network:
        rets.extend([(k, v, help_fmt % v) for k, v in NOVA_NETWORK_QUOTAS.items()])
    return rets