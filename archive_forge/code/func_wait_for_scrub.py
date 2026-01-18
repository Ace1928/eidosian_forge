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
def wait_for_scrub(self, image_id):
    """
        NOTE(jkoelker) The build servers sometimes take longer than 15 seconds
        to scrub. Give it up to 5 min, checking checking every 15 seconds.
        When/if it flips to deleted, bail immediately.
        """
    wait_for = 300
    check_every = 15
    for _ in range(wait_for // check_every):
        time.sleep(check_every)
        image = db_api.get_api().image_get(self.admin_context, image_id)
        if image['status'] == 'deleted' and image['deleted'] == True:
            break
        else:
            continue
    else:
        self.fail('image was never scrubbed')