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
class DomainConfigUpload(BaseApp):
    """Upload the domain specific configuration files to the database."""
    name = 'domain_config_upload'

    @classmethod
    def add_argument_parser(cls, subparsers):
        parser = super(DomainConfigUpload, cls).add_argument_parser(subparsers)
        parser.add_argument('--all', default=False, action='store_true', help='Upload contents of all domain specific configuration files. Either use this option or use the --domain-name option to choose a specific domain.')
        parser.add_argument('--domain-name', default=None, help='Upload contents of the specific configuration file for the given domain. Either use this option or use the --all option to upload contents for all domains.')
        return parser

    @staticmethod
    def main():
        dcu = DomainConfigUploadFiles()
        status = dcu.run()
        if status is not None:
            sys.exit(status)