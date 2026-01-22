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
def _get_datacenter(self, datacenter_path):
    search_index_moref = self.session.vim.service_content.searchIndex
    dc_moref = self.session.invoke_api(self.session.vim, 'FindByInventoryPath', search_index_moref, inventoryPath=datacenter_path)
    dc_name = datacenter_path.rsplit('/', 1)[-1]
    dc_obj = oslo_datacenter.Datacenter(ref=dc_moref, name=dc_name)
    dc_obj.path = datacenter_path
    return dc_obj