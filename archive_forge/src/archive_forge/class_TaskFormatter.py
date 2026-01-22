import logging
import os.path
from oslo_serialization import jsonutils
from osc_lib.command import command
from mistralclient.commands.v2 import base
from mistralclient.commands.v2 import executions
from mistralclient import utils
class TaskFormatter(base.MistralFormatter):
    COLUMNS = [('id', 'ID'), ('name', 'Name'), ('workflow_name', 'Workflow name'), ('workflow_namespace', 'Workflow namespace'), ('workflow_execution_id', 'Workflow Execution ID'), ('state', 'State'), ('state_info', 'State info'), ('created_at', 'Created at'), ('started_at', 'Started at'), ('finished_at', 'Finished at'), ('duration', 'Duration', True)]

    @staticmethod
    def format(task=None, lister=False):
        if task:
            state_info = task.state_info if not lister else base.cut(task.state_info)
            duration = base.get_duration_str(task.started_at, task.finished_at)
            data = (task.id, task.name, task.workflow_name, task.workflow_namespace, task.workflow_execution_id, task.state, state_info, task.created_at, task.started_at or '<none>', task.finished_at or '<none>', duration)
        else:
            data = (tuple(('' for _ in range(len(TaskFormatter.COLUMNS)))),)
        return (TaskFormatter.headings(), data)