import logging
from osc_lib.command import command
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
def _has_project_id(data):
    if len(data) < 1:
        return False
    if 'project_id' in data[0]:
        return True
    return False