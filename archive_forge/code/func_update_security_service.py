import ast
import re
import time
from oslo_utils import strutils
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import exceptions
from manilaclient.tests.functional import utils
@not_found_wrapper
def update_security_service(self, security_service, name=None, description=None, dns_ip=None, ou=None, server=None, domain=None, user=None, password=None, default_ad_site=None, microversion=None):
    cmd = 'security-service-update %s ' % security_service
    cmd += self._combine_security_service_data(name=name, description=description, dns_ip=dns_ip, ou=ou, server=server, domain=domain, user=user, password=password, default_ad_site=default_ad_site)
    return output_parser.details(self.manila(cmd, microversion=microversion))