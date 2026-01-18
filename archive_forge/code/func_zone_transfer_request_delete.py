import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def zone_transfer_request_delete(self, id, *args, **kwargs):
    cmd = f'zone transfer request delete {id}'
    return self.parsed_cmd(cmd, *args, **kwargs)