import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def zone_import_delete(self, zone_import_id, *args, **kwargs):
    cmd = f'zone import delete {zone_import_id}'
    return self.parsed_cmd(cmd, *args, **kwargs)