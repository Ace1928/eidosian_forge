import ast
import tenacity
from oslo_log import log as logging
from heat.common import exception
from heat.objects import sync_point as sync_point_object
def _str_unpack_tuple(s):
    s = s[s.index(':') + 1:]
    return ast.literal_eval(s)