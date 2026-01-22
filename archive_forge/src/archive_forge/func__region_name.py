import datetime
import warnings
from oslo_utils import timeutils
from keystoneclient.i18n import _
from keystoneclient import service_catalog
@property
def _region_name(self):
    return self.get('region_name')