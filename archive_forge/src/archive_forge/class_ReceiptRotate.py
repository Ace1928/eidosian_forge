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
class ReceiptRotate(BasePermissionsSetup):
    """Rotate auth receipts encryption keys.

    This assumes you have already run keystone-manage receipt_setup.

    A new primary key is placed into rotation, which is used for new receipts.
    The old primary key is demoted to secondary, which can then still be used
    for validating receipts. Excess secondary keys (beyond [receipt]
    max_active_keys) are revoked. Revoked keys are permanently deleted. A new
    staged key will be created and used to validate receipts. The next time key
    rotation takes place, the staged key will be put into rotation as the
    primary key.

    Rotating keys too frequently, or with [receipt] max_active_keys set
    too low, will cause receipts to become invalid prior to their expiration.

    """
    name = 'receipt_rotate'

    @classmethod
    def main(cls):
        keystone_user_id, keystone_group_id = cls.get_user_group()
        cls.rotate_fernet_repository(keystone_user_id, keystone_group_id, 'fernet_receipts')