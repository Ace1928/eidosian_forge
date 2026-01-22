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
class CredentialSetup(BasePermissionsSetup):
    """Setup a Fernet key repository for credential encryption.

    The purpose of this command is very similar to `keystone-manage
    fernet_setup` only the keys included in this repository are for encrypting
    and decrypting credential secrets instead of token payloads. Keys can be
    rotated using `keystone-manage credential_rotate`.
    """
    name = 'credential_setup'

    @classmethod
    def main(cls):
        futils = fernet_utils.FernetUtils(CONF.credential.key_repository, credential_fernet.MAX_ACTIVE_KEYS, 'credential')
        keystone_user_id, keystone_group_id = cls.get_user_group()
        futils.create_key_directory(keystone_user_id, keystone_group_id)
        if futils.validate_key_repository(requires_write=True):
            futils.initialize_key_repository(keystone_user_id, keystone_group_id)