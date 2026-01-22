import http.client
import eventlet
from oslo_serialization import jsonutils as json
from glance.api.v2 import tasks
from glance.common import timeutils
from glance.tests.integration.v2 import base
def _wait_on_task_execution(self, max_wait=5):
    """Wait until all the tasks have finished execution and are in
        state of success or failure.
        """
    start = timeutils.utcnow()
    while timeutils.delta_seconds(start, timeutils.utcnow()) < max_wait:
        wait = False
        path = '/v2/tasks'
        res, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        content_dict = json.loads(content)
        self.assertEqual(http.client.OK, res.status)
        res_tasks = content_dict['tasks']
        if len(res_tasks) != 0:
            for task in res_tasks:
                if task['status'] in ('pending', 'processing'):
                    wait = True
                    break
        if wait:
            eventlet.sleep(0.05)
            continue
        else:
            break