from oslo_config import cfg
from oslo_log import log as logging
from glance.api import versions
from glance.common import wsgi

        'Pops' off the next segment of PATH_INFO, returns the popped
        segment. Do NOT push it onto SCRIPT_NAME.
        