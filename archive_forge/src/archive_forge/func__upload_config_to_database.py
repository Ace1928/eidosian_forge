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
def _upload_config_to_database(self, file_name, domain_name):
    """Upload a single config file to the database.

        :param file_name: the file containing the config options
        :param domain_name: the domain name
        :returns: a boolean indicating if the upload succeeded

        """
    try:
        domain_ref = self.resource_manager.get_domain_by_name(domain_name)
    except exception.DomainNotFound:
        print(_('Invalid domain name: %(domain)s found in config file name: %(file)s - ignoring this file.') % {'domain': domain_name, 'file': file_name})
        return False
    if self.domain_config_manager.get_config_with_sensitive_info(domain_ref['id']):
        print(_('Domain: %(domain)s already has a configuration defined - ignoring file: %(file)s.') % {'domain': domain_name, 'file': file_name})
        return False
    sections = {}
    try:
        parser = cfg.ConfigParser(file_name, sections)
        parser.parse()
    except Exception:
        print(_('Error parsing configuration file for domain: %(domain)s, file: %(file)s.') % {'domain': domain_name, 'file': file_name})
        return False
    try:
        for group in sections:
            for option in sections[group]:
                sections[group][option] = sections[group][option][0]
        self.domain_config_manager.create_config(domain_ref['id'], sections)
        return True
    except Exception as e:
        msg = 'Error processing config file for domain: %(domain_name)s, file: %(filename)s, error: %(error)s'
        LOG.error(msg, {'domain_name': domain_name, 'filename': file_name, 'error': e}, exc_info=True)
        return False