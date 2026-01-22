import logging
from oslo_serialization import jsonutils
from osc_lib.command import command
from mistralclient.commands.v2 import base
from mistralclient import utils
class ActionExecutionFormatter(base.MistralFormatter):
    COLUMNS = [('id', 'ID'), ('name', 'Name'), ('workflow_name', 'Workflow name'), ('workflow_namespace', 'Workflow namespace'), ('task_name', 'Task name'), ('task_execution_id', 'Task ID'), ('state', 'State'), ('state_info', 'State info'), ('accepted', 'Accepted'), ('created_at', 'Created at'), ('updated_at', 'Updated at')]
    LIST_COLUMN_FIELD_NAMES = [c[0] for c in COLUMNS if c[0] != 'state_info']
    LIST_COLUMN_HEADING_NAMES = [c[1] for c in COLUMNS if c[0] != 'state_info']

    @staticmethod
    def format(action_ex=None, lister=False):
        if lister:
            columns = ActionExecutionFormatter.LIST_COLUMN_HEADING_NAMES
        else:
            columns = ActionExecutionFormatter.headings()
        if action_ex:
            if hasattr(action_ex, 'task_name'):
                task_name = action_ex.task_name
            else:
                task_name = None
            data = (action_ex.id, action_ex.name, action_ex.workflow_name, action_ex.workflow_namespace, task_name, action_ex.task_execution_id, action_ex.state)
            if not lister:
                data += (action_ex.state_info,)
            data += (action_ex.accepted, action_ex.created_at, action_ex.updated_at or '<none>')
        else:
            data = (('',) * len(columns),)
        return (columns, data)