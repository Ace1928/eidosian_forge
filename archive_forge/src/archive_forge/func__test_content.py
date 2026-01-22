import http.client
import os
import sys
import time
import httplib2
from oslo_config import cfg
from oslo_serialization import jsonutils
from oslo_utils.fixture import uuidsentinel as uuids
from glance import context
import glance.db as db_api
from glance.tests import functional
from glance.tests.utils import execute
def _test_content():
    scrubber = functional.ScrubberDaemon(self.test_dir, self.policy_file)
    scrubber.write_conf(daemon=False)
    scrubber.needs_database = True
    scrubber.create_database()
    exe_cmd = '%s -m glance.cmd.scrubber' % sys.executable
    cmd = '%s --config-file %s --restore fake_image_id' % (exe_cmd, scrubber.conf_file_name)
    return execute(cmd, raise_error=False)