import logging
import os
import urllib.parse
from oslo_config import cfg
from oslo_utils import excutils
from oslo_utils import netutils
from oslo_utils import units
import requests
from requests import adapters
from requests.packages.urllib3.util import retry
import glance_store
from glance_store import capabilities
from glance_store.common import utils
from glance_store import exceptions
from glance_store.i18n import _, _LE
from glance_store import location
def _get_freespace(self, ds_obj):
    return self.session.invoke_api(vim_util, 'get_object_property', self.session.vim, ds_obj.ref, 'summary.freeSpace')