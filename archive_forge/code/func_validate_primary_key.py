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
def validate_primary_key(self):
    crypto, keys = credential_fernet.get_multi_fernet_keys()
    primary_key_hash = credential_fernet.primary_key_hash(keys)
    credentials = self.credential_api.driver.list_credentials(driver_hints.Hints())
    for credential in credentials:
        if credential['key_hash'] != primary_key_hash:
            msg = _('Unable to rotate credential keys because not all credentials are encrypted with the primary key. Please make sure all credentials have been encrypted with the primary key using `keystone-manage credential_migrate`.')
            raise SystemExit(msg)