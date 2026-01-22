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
class TrustFlush(BaseApp):
    """Flush expired and non-expired soft deleted trusts from the backend."""
    name = 'trust_flush'

    @classmethod
    def add_argument_parser(cls, subparsers):
        parser = super(TrustFlush, cls).add_argument_parser(subparsers)
        parser.add_argument('--project-id', default=None, help='The id of the project of which the expired or non-expired soft-deleted trusts is to be purged')
        parser.add_argument('--trustor-user-id', default=None, help='The id of the trustor of which the expired or non-expired soft-deleted trusts is to be purged')
        parser.add_argument('--trustee-user-id', default=None, help='The id of the trustee of which the expired or non-expired soft-deleted trusts is to be purged')
        parser.add_argument('--date', default=datetime.datetime.utcnow(), help='The date of which the expired or non-expired soft-deleted trusts older than that will be purged. The format of the date to be "DD-MM-YYYY". If no date is supplied keystone-manage will use the system clock time at runtime')
        return parser

    @classmethod
    def main(cls):
        drivers = backends.load_backends()
        trust_manager = drivers['trust_api']
        if CONF.command.date:
            if not isinstance(CONF.command.date, datetime.datetime):
                try:
                    CONF.command.date = datetime.datetime.strptime(CONF.command.date, '%d-%m-%Y')
                except KeyError:
                    raise ValueError("'%s'Invalid input for date, should be DD-MM-YYYY", CONF.command.date)
            else:
                LOG.info('No date is supplied, keystone-manage will use the system clock time at runtime ')
        trust_manager.flush_expired_and_soft_deleted_trusts(project_id=CONF.command.project_id, trustor_user_id=CONF.command.trustor_user_id, trustee_user_id=CONF.command.trustee_user_id, date=CONF.command.date)