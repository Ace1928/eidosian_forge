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
class MappingPopulate(BaseApp):
    """Pre-populate entries from domain-specific backends.

    Running this command is not required. It should only be run right after
    the LDAP was configured, when many new users were added, or when
    "mapping_purge" is run.

    This command will take a while to run. It is perfectly fine for it to run
    more than several minutes.
    """
    name = 'mapping_populate'

    @classmethod
    def load_backends(cls):
        drivers = backends.load_backends()
        cls.identity_api = drivers['identity_api']
        cls.resource_api = drivers['resource_api']

    @classmethod
    def add_argument_parser(cls, subparsers):
        parser = super(MappingPopulate, cls).add_argument_parser(subparsers)
        parser.add_argument('--domain-name', default=None, required=True, help='Name of the domain configured to use domain-specific backend')
        return parser

    @classmethod
    def main(cls):
        """Process entries for id_mapping_api."""
        cls.load_backends()
        domain_name = CONF.command.domain_name
        try:
            domain_id = cls.resource_api.get_domain_by_name(domain_name)['id']
        except exception.DomainNotFound:
            print(_('Invalid domain name: %(domain)s') % {'domain': domain_name})
            return False
        cls.identity_api.list_users(domain_scope=domain_id)