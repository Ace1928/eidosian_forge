import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def tsigkey_delete(self, id, *args, **kwargs):
    return self.parsed_cmd(f'tsigkey delete {id}', *args, **kwargs)