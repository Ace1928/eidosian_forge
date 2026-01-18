import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def zone_blacklist_show(self, id, *args, **kwargs):
    cmd = f'zone blacklist show {id}'
    return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)