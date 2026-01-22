import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
class TSIGKeyCommands:

    def tsigkey_list(self, *args, **kwargs):
        return self.parsed_cmd('tsigkey list', ListModel, *args, **kwargs)

    def tsigkey_show(self, id, *args, **kwargs):
        return self.parsed_cmd(f'tsigkey show {id}', FieldValueModel, *args, **kwargs)

    def tsigkey_delete(self, id, *args, **kwargs):
        return self.parsed_cmd(f'tsigkey delete {id}', *args, **kwargs)

    def tsigkey_create(self, name, algorithm, secret, scope, resource_id, *args, **kwargs):
        options_str = build_option_string({'--name': name, '--algorithm': algorithm, '--secret': secret, '--scope': scope, '--resource-id': resource_id})
        cmd = f'tsigkey create {options_str}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)

    def tsigkey_set(self, id, name=None, algorithm=None, secret=None, scope=None, *args, **kwargs):
        options_str = build_option_string({'--name': name, '--algorithm': algorithm, '--secret': secret, '--scope': scope})
        cmd = f'tsigkey set {id} {options_str}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)