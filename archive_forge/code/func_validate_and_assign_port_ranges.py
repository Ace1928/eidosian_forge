import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
def validate_and_assign_port_ranges(parsed_args, attrs):
    internal_port_range = parsed_args.internal_protocol_port
    external_port_range = parsed_args.external_protocol_port
    external_ports = internal_ports = []
    if external_port_range:
        external_ports = list(map(int, str(external_port_range).split(':')))
    if internal_port_range:
        internal_ports = list(map(int, str(internal_port_range).split(':')))
    validate_ports_match(internal_ports, external_ports)
    for port in external_ports + internal_ports:
        validate_port(port)
    if internal_port_range:
        if ':' in internal_port_range:
            attrs['internal_port_range'] = internal_port_range
        else:
            attrs['internal_port'] = int(internal_port_range)
    if external_port_range:
        if ':' in external_port_range:
            attrs['external_port_range'] = external_port_range
        else:
            attrs['external_port'] = int(external_port_range)