import ast
import tenacity
from oslo_log import log as logging
from heat.common import exception
from heat.objects import sync_point as sync_point_object
def _dump_list(items, separator=', '):
    return separator.join(map(str, items))