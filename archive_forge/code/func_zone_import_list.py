import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def zone_import_list(self, *args, **kwargs):
    cmd = 'zone import list'
    return self.parsed_cmd(cmd, ListModel, *args, **kwargs)