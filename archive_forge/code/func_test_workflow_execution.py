import time
import uuid
import json
import traceback
from boto.swf.layer1_decisions import Layer1Decisions
from tests.integration.swf.test_layer1 import SimpleWorkflowLayer1TestBase
def test_workflow_execution(self):
    workflow_id = 'wfid-%.2f' % (time.time(),)
    r = self.conn.start_workflow_execution(self._domain, workflow_id, self._workflow_type_name, self._workflow_type_version, execution_start_to_close_timeout='20', input='[600, 15]')
    run_id = r['runId']
    self.run_decider()
    self.run_worker()
    self.run_decider()
    r = self.conn.get_workflow_execution_history(self._domain, run_id, workflow_id, reverse_order=True)['events'][0]
    result = r['workflowExecutionCompletedEventAttributes']['result']
    assert json.loads(result) == 615