import copy
import functools
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from glance.common import exception
from glance.common import timeutils
from glance.common import utils
from glance.db import utils as db_utils
from glance.i18n import _, _LI, _LW
def _metadef_delete_namespace_content(get_func, key, context, namespace_name):
    global DATA
    metadefs = get_func(context, namespace_name)
    data = DATA[key]
    for metadef in metadefs:
        data.remove(metadef)
    return metadefs