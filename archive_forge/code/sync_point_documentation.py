import ast
import tenacity
from oslo_log import log as logging
from heat.common import exception
from heat.objects import sync_point as sync_point_object
Deletes all sync points of a stack associated with a traversal_id.