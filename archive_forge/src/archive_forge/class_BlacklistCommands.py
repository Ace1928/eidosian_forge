import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
class BlacklistCommands:

    def zone_blacklist_list(self, *args, **kwargs):
        cmd = 'zone blacklist list'
        return self.parsed_cmd(cmd, ListModel, *args, **kwargs)

    def zone_blacklist_create(self, pattern, description=None, *args, **kwargs):
        options_str = build_option_string({'--pattern': pattern, '--description': description})
        cmd = f'zone blacklist create {options_str}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)

    def zone_blacklist_set(self, id, pattern=None, description=None, no_description=False, *args, **kwargs):
        options_str = build_option_string({'--pattern': pattern, '--description': description})
        flags_str = build_flags_string({'--no-description': no_description})
        cmd = f'zone blacklist set {id} {options_str} {flags_str}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)

    def zone_blacklist_show(self, id, *args, **kwargs):
        cmd = f'zone blacklist show {id}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)

    def zone_blacklist_delete(self, id, *args, **kwargs):
        cmd = f'zone blacklist delete {id}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)