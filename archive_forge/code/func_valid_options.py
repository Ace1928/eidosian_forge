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
def valid_options(self):
    """Validate the options, returning True if they are indeed valid.

        It would be nice to use the argparse automated checking for this
        validation, but the only way I can see doing that is to make the
        default (i.e. if no optional parameters are specified) to upload
        all configuration files - and that sounds too dangerous as a
        default. So we use it in a slightly unconventional way, where all
        parameters are optional, but you must specify at least one.

        """
    if CONF.command.all is False and CONF.command.domain_name is None:
        print(_('At least one option must be provided, use either --all or --domain-name'))
        return False
    if CONF.command.all is True and CONF.command.domain_name is not None:
        print(_('The --all option cannot be used with the --domain-name option'))
        return False
    return True