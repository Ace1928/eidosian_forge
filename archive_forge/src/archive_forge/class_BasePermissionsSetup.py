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
class BasePermissionsSetup(BaseApp):
    """Common user/group setup for file permissions."""

    @classmethod
    def add_argument_parser(cls, subparsers):
        parser = super(BasePermissionsSetup, cls).add_argument_parser(subparsers)
        running_as_root = os.geteuid() == 0
        parser.add_argument('--keystone-user', required=running_as_root)
        parser.add_argument('--keystone-group', required=running_as_root)
        return parser

    @staticmethod
    def get_user_group():
        keystone_user_id = None
        keystone_group_id = None
        try:
            a = CONF.command.keystone_user
            if a:
                keystone_user_id = utils.get_unix_user(a)[0]
        except KeyError:
            raise ValueError("Unknown user '%s' in --keystone-user" % a)
        try:
            a = CONF.command.keystone_group
            if a:
                keystone_group_id = utils.get_unix_group(a)[0]
        except KeyError:
            raise ValueError("Unknown group '%s' in --keystone-group" % a)
        return (keystone_user_id, keystone_group_id)

    @classmethod
    def initialize_fernet_repository(cls, keystone_user_id, keystone_group_id, config_group=None):
        conf_group = getattr(CONF, config_group)
        futils = fernet_utils.FernetUtils(conf_group.key_repository, conf_group.max_active_keys, config_group)
        futils.create_key_directory(keystone_user_id, keystone_group_id)
        if futils.validate_key_repository(requires_write=True):
            futils.initialize_key_repository(keystone_user_id, keystone_group_id)

    @classmethod
    def rotate_fernet_repository(cls, keystone_user_id, keystone_group_id, config_group=None):
        conf_group = getattr(CONF, config_group)
        futils = fernet_utils.FernetUtils(conf_group.key_repository, conf_group.max_active_keys, config_group)
        if futils.validate_key_repository(requires_write=True):
            futils.rotate_keys(keystone_user_id, keystone_group_id)