import sys
from oslo_config import cfg
from oslo_log import log
from heat.common import context
from heat.common import exception
from heat.common.i18n import _
from heat.common import messaging
from heat.common import service_utils
from heat.db import api as db_api
from heat.db import migration as db_migration
from heat.objects import service as service_objects
from heat.rpc import client as rpc_client
from heat import version
@staticmethod
def add_service_parsers(subparsers):
    service_parser = subparsers.add_parser('service')
    service_parser.set_defaults(command_object=ServiceManageCommand)
    service_subparsers = service_parser.add_subparsers(dest='action')
    list_parser = service_subparsers.add_parser('list')
    list_parser.set_defaults(func=ServiceManageCommand().service_list)
    remove_parser = service_subparsers.add_parser('clean')
    remove_parser.set_defaults(func=ServiceManageCommand().service_clean)