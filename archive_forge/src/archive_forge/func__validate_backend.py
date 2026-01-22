import json
import os
import sys
from oslo_utils import strutils
from glanceclient._i18n import _
from glanceclient.common import progressbar
from glanceclient.common import utils
from glanceclient import exc
from glanceclient.v2 import cache
from glanceclient.v2 import image_members
from glanceclient.v2 import image_schema
from glanceclient.v2 import images
from glanceclient.v2 import namespace_schema
from glanceclient.v2 import resource_type_schema
from glanceclient.v2 import tasks
def _validate_backend(backend, gc):
    try:
        enabled_backends = gc.images.get_stores_info().get('stores')
    except exc.HTTPNotFound:
        return
    if backend:
        valid_backend = False
        for available_backend in enabled_backends:
            if available_backend['id'] == backend:
                valid_backend = True
                break
        if not valid_backend:
            utils.exit("Store '%s' is not valid for this cloud. Valid values can be retrieved with stores-info command." % backend)