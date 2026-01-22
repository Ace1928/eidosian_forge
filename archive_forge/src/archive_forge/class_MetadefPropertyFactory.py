from collections import abc
import datetime
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import importutils
from glance.common import exception
from glance.common import timeutils
from glance.i18n import _, _LE, _LI, _LW
class MetadefPropertyFactory(object):

    def new_namespace_property(self, namespace, name, schema, **kwargs):
        property_id = str(uuid.uuid4())
        return MetadefProperty(namespace, property_id, name, schema)