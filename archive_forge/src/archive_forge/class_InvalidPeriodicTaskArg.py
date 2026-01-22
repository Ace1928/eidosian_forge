import copy
import logging
import random
import time
from time import monotonic as now
from oslo_service._i18n import _
from oslo_service import _options
from oslo_utils import reflection
class InvalidPeriodicTaskArg(Exception):
    message = _('Unexpected argument for periodic task creation: %(arg)s.')