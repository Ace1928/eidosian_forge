import copy
from oslo_utils import encodeutils
from oslo_utils import strutils
import urllib.parse
from glanceclient.common import utils
from glanceclient.v1.apiclient import base
@staticmethod
def _format_image_meta_for_user(meta):
    for key in ['size', 'min_ram', 'min_disk']:
        if key in meta:
            try:
                meta[key] = int(meta[key]) if meta[key] else 0
            except ValueError:
                pass
    return meta