import ast
import logging
from cliff import command
from cliff.formatters import table
from cliff import lister
from cliff import show
from blazarclient import exception
from blazarclient import utils
class BlazarCommand(OpenStackCommand):
    """Base Blazar CLI command."""
    api = 'reservation'
    log = logging.getLogger(__name__ + '.BlazarCommand')
    values_specs = []
    json_indent = None
    resource = None
    allow_names = True
    name_key = None
    id_pattern = UUID_PATTERN

    def __init__(self, app, app_args):
        super(BlazarCommand, self).__init__(app, app_args)

    def get_client(self):
        if hasattr(self.app, 'client_manager'):
            return self.app.client_manager.reservation
        else:
            return self.app.client

    def get_parser(self, prog_name):
        parser = super(BlazarCommand, self).get_parser(prog_name)
        return parser

    def format_output_data(self, data):
        for k, v in data.items():
            if isinstance(v, str):
                try:
                    v = ast.literal_eval(v)
                except SyntaxError:
                    pass
                except ValueError:
                    pass
            if isinstance(v, list):
                value = '\n'.join((utils.dumps(i, indent=self.json_indent) if isinstance(i, dict) else str(i) for i in v))
                data[k] = value
            elif isinstance(v, dict):
                value = utils.dumps(v, indent=self.json_indent)
                data[k] = value
            elif v is None:
                data[k] = ''

    def add_known_arguments(self, parser):
        pass

    def args2body(self, parsed_args):
        return {}