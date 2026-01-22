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
def _format_association(namespace, resource_type, association_values):
    association = {'namespace_id': namespace['id'], 'resource_type': resource_type['id'], 'properties_target': None, 'prefix': None, 'created_at': timeutils.utcnow(), 'updated_at': timeutils.utcnow()}
    association.update(association_values)
    return association