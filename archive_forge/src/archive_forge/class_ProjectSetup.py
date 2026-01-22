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
class ProjectSetup(BaseApp):
    """Create project with specified UUID."""
    name = 'project_setup'

    def __init__(self):
        self.identity = idutils.Identity()

    @classmethod
    def add_argument_parser(cls, subparsers):
        parser = super(ProjectSetup, cls).add_argument_parser(subparsers)
        parser.add_argument('--project-name', default=None, required=True, help='The name of the keystone project being created.')
        parser.add_argument('--project-id', default=None, help='The UUID of the keystone project being created.')
        return parser

    def do_project_setup(self):
        """Create project with specified UUID."""
        self.identity.project_name = CONF.command.project_name
        self.identity.project_id = CONF.command.project_id
        self.identity.project_setup()

    @classmethod
    def main(cls):
        klass = cls()
        klass.do_project_setup()