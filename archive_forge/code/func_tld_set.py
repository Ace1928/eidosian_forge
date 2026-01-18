import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def tld_set(self, id, name=None, description=None, no_description=False, *args, **kwargs):
    options_str = build_option_string({'--name': name, '--description': description})
    flags_str = build_flags_string({'--no-description': no_description})
    cmd = f'tld set {id} {options_str} {flags_str}'
    return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)