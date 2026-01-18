import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def zone_transfer_request_set(self, id, description=None, *args, **kwargs):
    options_str = build_option_string({'--description': description})
    cmd = f'zone transfer request set {options_str} {id}'
    return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)