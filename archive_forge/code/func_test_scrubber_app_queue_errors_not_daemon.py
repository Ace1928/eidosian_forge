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
def test_scrubber_app_queue_errors_not_daemon(self):
    """
        test that the glance-scrubber exits with an exit code > 0 when it
        fails to lookup images, indicating a configuration error when not
        in daemon mode.

        Related-Bug: #1548289
        """
    exitcode, out, err = self.scrubber_daemon.start(delayed_delete=True, daemon=False)
    self.assertEqual(0, exitcode, 'Failed to spin up the Scrubber daemon. Got: %s' % err)
    exe_cmd = '%s -m glance.cmd.scrubber' % sys.executable
    cmd = '%s --config-file %s' % (exe_cmd, self.scrubber_daemon.conf_file_name)
    exitcode, out, err = execute(cmd, raise_error=False)
    self.assertEqual(1, exitcode)
    self.assertIn('Can not get scrub jobs from queue', str(err))
    self.stop_server(self.scrubber_daemon)