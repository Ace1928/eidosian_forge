import logging
from osc_lib.command import command
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2 import utils as v2_utils
def _format_status(status):
    status.pop('links', None)
    for k in ('capabilities', 'stats'):
        status[k] = '\n'.join(status[k]) if status[k] else '-'
    return status