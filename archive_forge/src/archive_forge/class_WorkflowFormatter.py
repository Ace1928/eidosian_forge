import argparse
from cliff import command
from cliff import show
from mistralclient.commands.v2 import base
from mistralclient import utils
class WorkflowFormatter(base.MistralFormatter):
    COLUMNS = [('id', 'ID'), ('name', 'Name'), ('namespace', 'Namespace'), ('project_id', 'Project ID'), ('tags', 'Tags'), ('input', 'Input'), ('scope', 'Scope'), ('created_at', 'Created at'), ('updated_at', 'Updated at')]

    @staticmethod
    def format(workflow=None, lister=False):
        if workflow:
            tags = getattr(workflow, 'tags', None) or []
            data = (workflow.id, workflow.name, workflow.namespace, workflow.project_id, base.wrap(', '.join(tags)) or '<none>', workflow.input if not lister else base.cut(workflow.input), workflow.scope, workflow.created_at)
            if hasattr(workflow, 'updated_at'):
                data += (workflow.updated_at,)
            else:
                data += (None,)
        else:
            data = (tuple(('' for _ in range(len(WorkflowFormatter.COLUMNS)))),)
        return (WorkflowFormatter.headings(), data)