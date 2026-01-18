import time
from functools import wraps
from boto.swf.layer1 import Layer1
from boto.swf.layer1_decisions import Layer1Decisions
@wraps(Layer1.list_workflow_types)
def workflows(self, status='REGISTERED', **kwargs):
    """ListWorkflowTypes."""
    wf_types = self._swf.list_workflow_types(self.name, status, **kwargs)
    wf_objects = []
    for wf_args in wf_types['typeInfos']:
        wf_ident = wf_args['workflowType']
        del wf_args['workflowType']
        wf_args.update(wf_ident)
        wf_args.update({'aws_access_key_id': self.aws_access_key_id, 'aws_secret_access_key': self.aws_secret_access_key, 'domain': self.name, 'region': self.region})
        wf_objects.append(WorkflowType(**wf_args))
    return wf_objects