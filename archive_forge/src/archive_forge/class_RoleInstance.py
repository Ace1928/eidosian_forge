import re
import copy
import time
import base64
import random
import collections
from xml.dom import minidom
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape
from libcloud.utils.py3 import ET, httplib, urlparse
from libcloud.utils.py3 import urlquote as url_quote
from libcloud.utils.py3 import _real_unicode, ensure_string
from libcloud.utils.misc import ReprMixin
from libcloud.common.azure import AzureRedirectException, AzureServiceManagementConnection
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState
from libcloud.compute.providers import Provider
class RoleInstance(WindowsAzureData):

    def __init__(self):
        self.role_name = ''
        self.instance_name = ''
        self.instance_status = ''
        self.instance_upgrade_domain = 0
        self.instance_fault_domain = 0
        self.instance_size = ''
        self.instance_state_details = ''
        self.instance_error_code = ''
        self.ip_address = ''
        self.instance_endpoints = InstanceEndpoints()
        self.power_state = ''
        self.fqdn = ''
        self.host_name = ''