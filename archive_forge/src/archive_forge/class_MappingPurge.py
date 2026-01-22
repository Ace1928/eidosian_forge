import argparse
import datetime
import os
import sys
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exception
from oslo_log import log
from oslo_serialization import jsonutils
import pbr.version
from keystone.cmd import bootstrap
from keystone.cmd import doctor
from keystone.cmd import idutils
from keystone.common import driver_hints
from keystone.common import fernet_utils
from keystone.common import jwt_utils
from keystone.common import sql
from keystone.common.sql import upgrades
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.federation import idp
from keystone.federation import utils as mapping_engine
from keystone.i18n import _
from keystone.server import backends
class MappingPurge(BaseApp):
    """Purge the mapping table."""
    name = 'mapping_purge'

    @classmethod
    def add_argument_parser(cls, subparsers):
        parser = super(MappingPurge, cls).add_argument_parser(subparsers)
        parser.add_argument('--all', default=False, action='store_true', help='Purge all mappings.')
        parser.add_argument('--domain-name', default=None, help='Purge any mappings for the domain specified.')
        parser.add_argument('--public-id', default=None, help='Purge the mapping for the Public ID specified.')
        parser.add_argument('--local-id', default=None, help='Purge the mappings for the Local ID specified.')
        parser.add_argument('--type', default=None, choices=['user', 'group'], help='Purge any mappings for the type specified.')
        return parser

    @staticmethod
    def main():

        def validate_options():
            if CONF.command.all is False and CONF.command.domain_name is None and (CONF.command.public_id is None) and (CONF.command.local_id is None) and (CONF.command.type is None):
                raise ValueError(_('At least one option must be provided'))
            if CONF.command.all is True and (CONF.command.domain_name is not None or CONF.command.public_id is not None or CONF.command.local_id is not None or (CONF.command.type is not None)):
                raise ValueError(_('--all option cannot be mixed with other options'))

        def get_domain_id(name):
            try:
                return resource_manager.get_domain_by_name(name)['id']
            except KeyError:
                raise ValueError(_("Unknown domain '%(name)s' specified by --domain-name") % {'name': name})
        validate_options()
        drivers = backends.load_backends()
        resource_manager = drivers['resource_api']
        mapping_manager = drivers['id_mapping_api']
        mapping = {}
        if CONF.command.domain_name is not None:
            mapping['domain_id'] = get_domain_id(CONF.command.domain_name)
        if CONF.command.public_id is not None:
            mapping['public_id'] = CONF.command.public_id
        if CONF.command.local_id is not None:
            mapping['local_id'] = CONF.command.local_id
        if CONF.command.type is not None:
            mapping['entity_type'] = CONF.command.type
        mapping_manager.purge_mappings(mapping)