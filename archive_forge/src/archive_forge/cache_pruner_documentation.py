import os
import sys
from oslo_log import log as logging
from glance.common import config
from glance.image_cache import pruner

Glance Image Cache Pruner

This is meant to be run as a periodic task, perhaps every half-hour.
