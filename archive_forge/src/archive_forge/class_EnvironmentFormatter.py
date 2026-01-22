import argparse
from oslo_serialization import jsonutils
from osc_lib.command import command
from mistralclient.commands.v2 import base
from mistralclient import utils
class EnvironmentFormatter(base.MistralFormatter):
    COLUMNS = [('name', 'Name'), ('description', 'Description'), ('variables', 'Variables'), ('scope', 'Scope'), ('created_at', 'Created at'), ('updated_at', 'Updated at')]
    LIST_COLUMN_FIELD_NAMES = [c[0] for c in COLUMNS if c[0] != 'variables']
    LIST_COLUMN_HEADING_NAMES = [c[1] for c in COLUMNS if c[0] != 'variables']

    @staticmethod
    def format(environment=None, lister=False):
        if lister:
            columns = EnvironmentFormatter.LIST_COLUMN_HEADING_NAMES
        else:
            columns = EnvironmentFormatter.headings()
        if environment:
            data = (environment.name,)
            if hasattr(environment, 'description'):
                data += (environment.description or '<none>',)
            else:
                data += (None,)
            if not lister:
                data += (jsonutils.dumps(environment.variables, indent=4),)
            data += (environment.scope, environment.created_at)
            if hasattr(environment, 'updated_at'):
                data += (environment.updated_at or '<none>',)
            else:
                data += (None,)
        else:
            data = (tuple(('' for _ in range(len(columns)))),)
        return (columns, data)