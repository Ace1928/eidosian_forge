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
def _check_readonly(self, kwargs):
    for key in self._readonly_properties:
        if key in kwargs:
            raise exception.ReadonlyProperty(property=key)