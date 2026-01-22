import os
import sys
import glance_store
from oslo_log import log as logging
from glance.common import config
from glance.image_cache import prefetcher

Glance Image Cache Pre-fetcher

This is meant to be run from the command line after queueing
images to be pretched.
