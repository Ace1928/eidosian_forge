import functools
import inspect
import logging
from oslo_config import cfg
from oslo_utils import excutils
import webob.exc
from oslo_versionedobjects._i18n import _
class EnumRequiresValidValuesError(VersionedObjectsException):
    msg_fmt = _('Enum fields require a list of valid_values')