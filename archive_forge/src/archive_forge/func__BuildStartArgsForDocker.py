from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import os
from googlecloudsdk.command_lib.emulators import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
import six
def _BuildStartArgsForDocker(args):
    """Builds arguments for starting the spanner emulator under docker."""
    host_ip = args.host_port.host
    if host_ip == 'localhost':
        host_ip = '127.0.0.1'
    try:
        ipaddress.ip_address(host_ip)
    except ValueError:
        raise InvalidHostPortFormat('When using docker, hostname specified via --host-port must be an IPV4 or IPV6 address, found ' + host_ip)
    if args.enable_fault_injection:
        return execution_utils.ArgsForExecutableTool('docker', 'run', '-p', '{}:{}:{}'.format(host_ip, args.host_port.port, SPANNER_EMULATOR_DEFAULT_GRPC_PORT), '-p', '{}:{}:{}'.format(host_ip, args.rest_port, SPANNER_EMULATOR_DEFAULT_REST_PORT), SPANNER_EMULATOR_DOCKER_IMAGE, './gateway_main', '--hostname', '0.0.0.0', '--enable_fault_injection')
    else:
        return execution_utils.ArgsForExecutableTool('docker', 'run', '-p', '{}:{}:{}'.format(host_ip, args.host_port.port, SPANNER_EMULATOR_DEFAULT_GRPC_PORT), '-p', '{}:{}:{}'.format(host_ip, args.rest_port, SPANNER_EMULATOR_DEFAULT_REST_PORT), SPANNER_EMULATOR_DOCKER_IMAGE)