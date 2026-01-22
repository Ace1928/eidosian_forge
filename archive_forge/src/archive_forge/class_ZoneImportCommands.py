import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
class ZoneImportCommands:
    """A mixin for DesignateCLI to add zone import commands"""

    def zone_import_list(self, *args, **kwargs):
        cmd = 'zone import list'
        return self.parsed_cmd(cmd, ListModel, *args, **kwargs)

    def zone_import_create(self, zone_file_path, *args, **kwargs):
        cmd = f'zone import create {zone_file_path}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)

    def zone_import_show(self, zone_import_id, *args, **kwargs):
        cmd = f'zone import show {zone_import_id}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)

    def zone_import_delete(self, zone_import_id, *args, **kwargs):
        cmd = f'zone import delete {zone_import_id}'
        return self.parsed_cmd(cmd, *args, **kwargs)