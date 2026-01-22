from typing import Optional, Union
from .helpers import LineKey, PCColumn
from .util import Attr, Panel, coalesce, nested_get, nested_set
from .validators import (
class WeavePanelSummaryTable(Panel):
    table_name: Optional[str] = Attr(json_path='spec.config.panel2Config.exp.fromOp.inputs.key.val')

    def __init__(self, table_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spec['config'] = self._default_config()
        self.table_name = table_name

    @classmethod
    def from_json(cls, spec):
        table_name = spec['config']['panel2Config']['exp']['fromOp']['inputs']['key']['val']
        return cls(table_name)

    @property
    def view_type(self) -> str:
        return 'Weave'

    @staticmethod
    def _default_config():
        return {'panel2Config': {'exp': {'nodeType': 'output', 'type': {'type': 'tagged', 'tag': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'entityName': 'string', 'projectName': 'string'}}, 'value': {'type': 'typedDict', 'propertyTypes': {'project': 'project', 'filter': 'string', 'order': 'string'}}}, 'value': {'type': 'list', 'objectType': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'run': 'run'}}, 'value': {'type': 'union', 'members': [{'type': 'file', 'extension': 'json', 'wbObjectType': {'type': 'table', 'columnTypes': {}}}, 'none']}}, 'maxLength': 50}}, 'fromOp': {'name': 'pick', 'inputs': {'obj': {'nodeType': 'output', 'type': {'type': 'tagged', 'tag': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'entityName': 'string', 'projectName': 'string'}}, 'value': {'type': 'typedDict', 'propertyTypes': {'project': 'project', 'filter': 'string', 'order': 'string'}}}, 'value': {'type': 'list', 'objectType': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'run': 'run'}}, 'value': {'type': 'union', 'members': [{'type': 'typedDict', 'propertyTypes': {'_wandb': {'type': 'typedDict', 'propertyTypes': {'runtime': 'number'}}}}, {'type': 'typedDict', 'propertyTypes': {'_step': 'number', 'table': {'type': 'file', 'extension': 'json', 'wbObjectType': {'type': 'table', 'columnTypes': {}}}, '_wandb': {'type': 'typedDict', 'propertyTypes': {'runtime': 'number'}}, '_runtime': 'number', '_timestamp': 'number'}}]}}, 'maxLength': 50}}, 'fromOp': {'name': 'run-summary', 'inputs': {'run': {'nodeType': 'var', 'type': {'type': 'tagged', 'tag': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'entityName': 'string', 'projectName': 'string'}}, 'value': {'type': 'typedDict', 'propertyTypes': {'project': 'project', 'filter': 'string', 'order': 'string'}}}, 'value': {'type': 'list', 'objectType': 'run', 'maxLength': 50}}, 'varName': 'runs'}}}}, 'key': {'nodeType': 'const', 'type': 'string', 'val': ''}}}, '__userInput': True}}}