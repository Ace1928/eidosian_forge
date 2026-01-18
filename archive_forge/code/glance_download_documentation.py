import urllib.request
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from taskflow.patterns import linear_flow as lf
from glance.async_.flows._internal_plugins import base_download
from glance.async_ import utils
from glance.common import exception
from glance.common import utils as common_utils
from glance.i18n import _, _LI, _LE
Create temp file into store and return path to it

        :param image_size: Glance Image Size retrieved from ImportMetadata task
        