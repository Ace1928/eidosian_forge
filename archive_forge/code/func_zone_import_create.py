import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def zone_import_create(self, zone_file_path, *args, **kwargs):
    cmd = f'zone import create {zone_file_path}'
    return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)