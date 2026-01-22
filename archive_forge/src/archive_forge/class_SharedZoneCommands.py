import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
class SharedZoneCommands:

    def shared_zone_show(self, zone_id, shared_zone_id, *args, **kwargs):
        cmd = f'zone share show {zone_id} {shared_zone_id}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)

    def shared_zone_list(self, zone_id, *args, **kwargs):
        cmd = f'zone share list {zone_id}'
        return self.parsed_cmd(cmd, ListModel, *args, **kwargs)

    def share_zone(self, zone_id, target_project_id, *args, **kwargs):
        cmd = f'zone share create {zone_id} {target_project_id}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)

    def unshare_zone(self, zone_id, shared_zone_id, *args, **kwargs):
        cmd = f'zone share delete {zone_id} {shared_zone_id}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)