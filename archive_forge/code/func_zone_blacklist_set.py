import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def zone_blacklist_set(self, id, pattern=None, description=None, no_description=False, *args, **kwargs):
    options_str = build_option_string({'--pattern': pattern, '--description': description})
    flags_str = build_flags_string({'--no-description': no_description})
    cmd = f'zone blacklist set {id} {options_str} {flags_str}'
    return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)