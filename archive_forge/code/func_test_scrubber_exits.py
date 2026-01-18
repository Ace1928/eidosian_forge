from unittest import mock
from unittest.mock import patch
import uuid
import glance_store
from oslo_config import cfg
from glance.common import exception
from glance.db.sqlalchemy import api as db_api
from glance import scrubber
from glance.tests import utils as test_utils
def test_scrubber_exits(self):
    scrub_jobs = scrubber.ScrubDBQueue.get_all_locations
    scrub_jobs = mock.MagicMock()
    scrub_jobs.side_effect = exception.NotFound
    scrub = scrubber.Scrubber(glance_store)
    self.assertRaises(exception.FailedToGetScrubberJobs, scrub._get_delete_jobs)