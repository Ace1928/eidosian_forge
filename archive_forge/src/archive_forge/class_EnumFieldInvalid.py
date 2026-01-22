import functools
import inspect
import logging
from oslo_config import cfg
from oslo_utils import excutils
import webob.exc
from oslo_versionedobjects._i18n import _
class EnumFieldInvalid(VersionedObjectsException):
    msg_fmt = _('%(typename)s in %(fieldname)s is not an instance of Enum')