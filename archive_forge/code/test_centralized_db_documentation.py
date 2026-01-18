import datetime
import errno
import io
import os
import time
from unittest import mock
from oslo_utils import fileutils
from glance.image_cache.drivers import centralized_db
from glance.tests import functional

        Test to see if open_for_write works in a failure case for each driver
        This case is where an exception is raised while the file is being
        written. The image is partially filled in cache and filling won't
        resume so verify the image is moved to invalid/ directory
        