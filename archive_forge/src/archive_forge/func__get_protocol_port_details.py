import argparse
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.common import exceptions as nc_exc
def _get_protocol_port_details(self, data, val):
    type_ip_prefix = val + '_ip_prefix'
    ip_prefix = data.get(type_ip_prefix)
    if not ip_prefix:
        ip_prefix = 'any'
    min_port = data.get(val + '_port_range_min')
    if min_port is None:
        min_port = 'any'
    max_port = data.get(val + '_port_range_max')
    if max_port is None:
        max_port = 'any'
    return '%s[port]: %s[%s:%s]' % (val, ip_prefix, min_port, max_port)