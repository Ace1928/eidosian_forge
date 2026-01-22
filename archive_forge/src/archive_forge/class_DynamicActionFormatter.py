from osc_lib.command import command
from mistralclient.commands.v2 import base
from mistralclient import utils
class DynamicActionFormatter(base.MistralFormatter):
    COLUMNS = [('id', 'ID'), ('name', 'Name'), ('class_name', 'Class'), ('code_source_id', 'Code source ID'), ('code_source_name', 'Code source name'), ('project_id', 'Project ID'), ('scope', 'Scope'), ('created_at', 'Created at'), ('updated_at', 'Updated at')]

    @staticmethod
    def format(action=None, lister=False):
        if action:
            data = (action.id, action.name, action.class_name, action.code_source_id, action.code_source_name, action.project_id, action.scope, action.created_at)
            if hasattr(action, 'updated_at'):
                data += (action.updated_at,)
            else:
                data += (None,)
        else:
            data = (('',) * len(DynamicActionFormatter.COLUMNS),)
        return (DynamicActionFormatter.headings(), data)