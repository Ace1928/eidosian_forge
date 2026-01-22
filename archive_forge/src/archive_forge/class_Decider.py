import time
from functools import wraps
from boto.swf.layer1 import Layer1
from boto.swf.layer1_decisions import Layer1Decisions
class Decider(Actor):
    """Base class for SimpleWorkflow deciders."""

    @wraps(Layer1.respond_decision_task_completed)
    def complete(self, task_token=None, decisions=None, **kwargs):
        """RespondDecisionTaskCompleted."""
        if isinstance(decisions, Layer1Decisions):
            decisions = decisions._data
        if task_token is None:
            task_token = self.last_tasktoken
        return self._swf.respond_decision_task_completed(task_token, decisions, **kwargs)

    @wraps(Layer1.poll_for_decision_task)
    def poll(self, **kwargs):
        """PollForDecisionTask."""
        task_list = self.task_list
        if 'task_list' in kwargs:
            task_list = kwargs.get('task_list')
            del kwargs['task_list']
        decision_task = self._swf.poll_for_decision_task(self.domain, task_list, **kwargs)
        self.last_tasktoken = decision_task.get('taskToken')
        return decision_task