import datetime
from testtools import content as ttc
import time
from unittest import mock
import uuid
from oslo_log import log as logging
from oslo_utils import fixture as time_fixture
from oslo_utils import units
from glance.tests import functional
from glance.tests import utils as test_utils
def _test_import_copy(self, warp_time=False):
    self.start_server()
    state = {'want_run': True}
    image_id = self._create_and_import(stores=['store1'])

    def slow_fake_set_data(data_iter, size=None, backend=None, set_active=True):
        me = str(uuid.uuid4())
        while state['want_run'] == True:
            LOG.info('fake_set_data running %s', me)
            state['running'] = True
            time.sleep(0.1)
        LOG.info('fake_set_data ended %s', me)
    tf = time_fixture.TimeFixture()
    self.useFixture(tf)
    with mock.patch('glance.location.ImageProxy.set_data') as mock_sd:
        mock_sd.side_effect = slow_fake_set_data
        resp = self._import_copy(image_id, ['store2'])
        self.addDetail('First import response', ttc.text_content(str(resp)))
        self.assertEqual(202, resp.status_code)
        for i in range(0, 10):
            if 'running' in state:
                break
            time.sleep(0.1)
    self.assertTrue(state.get('running', False), 'slow_fake_set_data() never ran')
    first_import_task = self._get_image_import_task(image_id)
    self.assertEqual('processing', first_import_task['status'])
    if warp_time:
        tf.advance_time_delta(datetime.timedelta(hours=2))
    resp = self._import_copy(image_id, ['store3'])
    time.sleep(0.1)
    self.addDetail('Second import response', ttc.text_content(str(resp)))
    if warp_time:
        self.assertEqual(202, resp.status_code)
    else:
        self.assertEqual(409, resp.status_code)
    self.addDetail('First task', ttc.text_content(str(first_import_task)))
    second_import_task = self._get_image_import_task(image_id)
    first_import_task = self._get_image_import_task(image_id, first_import_task['id'])
    if warp_time:
        self.assertNotEqual(first_import_task['id'], second_import_task['id'])
        self.assertEqual('failure', first_import_task['status'])
        self.assertEqual('Expired lock preempted', first_import_task['message'])
        self.assertEqual('processing', second_import_task['status'])
    else:
        self.assertEqual(first_import_task['id'], second_import_task['id'])
    return (image_id, state)