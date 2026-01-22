import argparse
from contextlib import closing
import io
import os
from oslo_log import log as logging
import tarfile
import time
from osc_lib.command import command
from osc_lib import utils
from zunclient.common import utils as zun_utils
from zunclient.common.websocketclient import exceptions
from zunclient.common.websocketclient import websocketclient
from zunclient import exceptions as exc
from zunclient.i18n import _
class RunContainer(command.ShowOne):
    """Create and run a new container"""
    log = logging.getLogger(__name__ + '.RunContainer')

    def get_parser(self, prog_name):
        parser = super(RunContainer, self).get_parser(prog_name)
        parser.add_argument('--name', metavar='<name>', help='name of the container')
        parser.add_argument('image', metavar='<image>', help='name or ID of the image')
        parser.add_argument('--cpu', metavar='<cpu>', help='The number of virtual cpus.')
        parser.add_argument('--memory', metavar='<memory>', help='The container memory size in MiB')
        parser.add_argument('--environment', metavar='<KEY=VALUE>', action='append', default=[], help='The environment variables')
        parser.add_argument('--workdir', metavar='<workdir>', help='The working directory for commands to run in')
        parser.add_argument('--label', metavar='<KEY=VALUE>', action='append', default=[], help='Adds a map of labels to a container. May be used multiple times.')
        parser.add_argument('--image-pull-policy', dest='image_pull_policy', metavar='<policy>', choices=['never', 'always', 'ifnotpresent'], help='The policy which determines if the image should be pulled prior to starting the container. It can have following values: "ifnotpresent": only pull the image if it does not already exist on the node. "always": Always pull the image from repository."never": never pull the image')
        restart_auto_remove_args = parser.add_mutually_exclusive_group()
        restart_auto_remove_args.add_argument('--restart', metavar='<restart>', help='Restart policy to apply when a container exits(no, on-failure[:max-retry], always, unless-stopped)')
        restart_auto_remove_args.add_argument('--auto-remove', dest='auto_remove', action='store_true', default=False, help='Automatically remove the container when it exits')
        parser.add_argument('--image-driver', metavar='<image_driver>', help='The image driver to use to pull container image. It can have following values: "docker": pull the image from Docker Hub. "glance": pull the image from Glance. ')
        parser.add_argument('--interactive', dest='interactive', action='store_true', default=False, help='Keep STDIN open even if not attached, allocate a pseudo-TTY')
        secgroup_expose_port_args = parser.add_mutually_exclusive_group()
        secgroup_expose_port_args.add_argument('--security-group', metavar='<security_group>', action='append', default=[], help='The name of security group for the container. May be used multiple times.')
        secgroup_expose_port_args.add_argument('--expose-port', action='append', default=[], metavar='<port>', help='Expose container port(s) to outside (format: <port>[/<protocol>]).')
        parser.add_argument('command', metavar='<command>', nargs=argparse.REMAINDER, help='Send command to the container')
        parser.add_argument('--hint', metavar='<key=value>', action='append', default=[], help='The key-value pair(s) for scheduler to select host. The format of this parameter is "key=value[,key=value]". May be used multiple times.')
        parser.add_argument('--net', metavar='<network=network, port=port-uuid,v4-fixed-ip=ip-addr,v6-fixed-ip=ip-addr>', action='append', default=[], help='Create network enpoints for the container. network: attach container to the specified neutron networks. port: attach container to the neutron port with this UUID. v4-fixed-ip: IPv4 fixed address for container. v6-fixed-ip: IPv6 fixed address for container.')
        parser.add_argument('--mount', action='append', default=[], metavar='<mount>', help='A dictionary to configure volumes mounted inside the container.')
        parser.add_argument('--runtime', metavar='<runtime>', help='The runtime to use for this container. It can have value "runc" or any other custom runtime.')
        parser.add_argument('--hostname', metavar='<hostname>', help='Container host name')
        parser.add_argument('--disk', metavar='<disk>', type=int, default=None, help='The disk size in GiB for per container.')
        parser.add_argument('--availability-zone', metavar='<availability_zone>', default=None, help='The availability zone of the container.')
        parser.add_argument('--auto-heal', dest='auto_heal', action='store_true', default=False, help='The flag of healing non-existent container in docker')
        parser.add_argument('--privileged', dest='privileged', action='store_true', default=False, help='Give extended privileges to this container')
        parser.add_argument('--healthcheck', action='append', default=[], metavar='<cmd=test_cmd,interval=time,retries=n,timeout=time>', help='Specify a test cmd to perform to check that the containeris healthy. cmd: Command to run to check health. interval: Time between running the check (``s|m|h``)          (default 0s). retries: Consecutive failures needed to report unhealthy.timeout: Maximum time to allow one check to run (``s|m|h``)         (default 0s).')
        parser.add_argument('--wait', action='store_true', help='Wait for run to complete')
        parser.add_argument('--registry', metavar='<registry>', help='The container image registry ID or name.')
        parser.add_argument('--host', metavar='<host>', help='Requested host to run containers. Admin only by default. (supported by --os-container-api-version 1.39 or above')
        parser.add_argument('--entrypoint', metavar='<entrypoint>', help='The entrypoint which overwrites the default ENTRYPOINT of the image.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        opts = {}
        opts['name'] = parsed_args.name
        opts['image'] = parsed_args.image
        opts['memory'] = parsed_args.memory
        opts['cpu'] = parsed_args.cpu
        opts['environment'] = zun_utils.format_args(parsed_args.environment)
        opts['workdir'] = parsed_args.workdir
        opts['labels'] = zun_utils.format_args(parsed_args.label)
        opts['image_pull_policy'] = parsed_args.image_pull_policy
        opts['image_driver'] = parsed_args.image_driver
        opts['auto_remove'] = parsed_args.auto_remove
        opts['command'] = parsed_args.command
        opts['registry'] = parsed_args.registry
        opts['host'] = parsed_args.host
        if parsed_args.entrypoint:
            opts['entrypoint'] = zun_utils.parse_entrypoint(parsed_args.entrypoint)
        if parsed_args.security_group:
            opts['security_groups'] = parsed_args.security_group
        if parsed_args.expose_port:
            opts['exposed_ports'] = zun_utils.parse_exposed_ports(parsed_args.expose_port)
        if parsed_args.restart:
            opts['restart_policy'] = zun_utils.check_restart_policy(parsed_args.restart)
        if parsed_args.interactive:
            opts['interactive'] = True
        if parsed_args.privileged:
            opts['privileged'] = True
        opts['hints'] = zun_utils.format_args(parsed_args.hint)
        opts['nets'] = zun_utils.parse_nets(parsed_args.net)
        opts['mounts'] = zun_utils.parse_mounts(parsed_args.mount)
        opts['runtime'] = parsed_args.runtime
        opts['hostname'] = parsed_args.hostname
        opts['disk'] = parsed_args.disk
        opts['availability_zone'] = parsed_args.availability_zone
        if parsed_args.auto_heal:
            opts['auto_heal'] = parsed_args.auto_heal
        if parsed_args.healthcheck:
            opts['healthcheck'] = zun_utils.parse_health(parsed_args.healthcheck)
        opts = zun_utils.remove_null_parms(**opts)
        container = client.containers.run(**opts)
        columns = _container_columns(container)
        container_uuid = getattr(container, 'uuid', None)
        if parsed_args.wait:
            if utils.wait_for_status(client.containers.get, container_uuid, success_status=['running']):
                container = client.containers.get(container_uuid)
            else:
                print('Failed to run container.\n')
                raise SystemExit
        if parsed_args.interactive:
            ready_for_attach = False
            while True:
                container = client.containers.get(container_uuid)
                if zun_utils.check_container_status(container, 'Running'):
                    ready_for_attach = True
                    break
                if zun_utils.check_container_status(container, 'Error'):
                    break
                print('Waiting for container start')
                time.sleep(1)
            if ready_for_attach is True:
                response = client.containers.attach(container_uuid)
                websocketclient.do_attach(client, response, container_uuid, '~', 0.5)
            else:
                raise exceptions.InvalidWebSocketLink(container_uuid)
        return (columns, utils.get_item_properties(container, columns))