import functools
import inspect
import logging
from oslo_config import cfg
from oslo_utils import excutils
import webob.exc
from oslo_versionedobjects._i18n import _
class UnregisteredSubobject(VersionedObjectsException):
    msg_fmt = _('%(child_objname)s is referenced by %(parent_objname)s but is not registered')