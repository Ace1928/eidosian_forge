import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def tld_create(self, name, description=None, *args, **kwargs):
    options_str = build_option_string({'--name': name, '--description': description})
    cmd = f'tld create {options_str}'
    return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)