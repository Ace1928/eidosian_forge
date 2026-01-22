import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
def _get_networks(opts_str):
    nic_args_list, opts_str = _strip_option(opts_str, 'nic', is_required=False, quotes_required=True, allow_multiple=True)
    nic_info_list = []
    for nic_args in nic_args_list:
        orig_nic_args = nic_args = _unquote(nic_args)
        nic_info = {}
        net_id, nic_args = _strip_option(nic_args, 'net-id', False)
        port_id, nic_args = _strip_option(nic_args, 'port-id', False)
        fixed_ipv4, nic_args = _strip_option(nic_args, 'v4-fixed-ip', False)
        if nic_args:
            raise exceptions.ValidationError(_("Unknown args '%s' in 'nic' option") % nic_args)
        if net_id:
            nic_info.update({'net-id': net_id})
        if port_id:
            nic_info.update({'port-id': port_id})
        if fixed_ipv4:
            nic_info.update({'v4-fixed-ip': fixed_ipv4})
        _validate_nic_info(nic_info, orig_nic_args)
        nic_info_list.append(nic_info)
    return (nic_info_list, opts_str)