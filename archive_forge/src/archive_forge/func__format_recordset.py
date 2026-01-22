import logging
from osc_lib.command import command
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
def _format_recordset(recordset):
    recordset['records'] = '\n'.join(recordset['records'])
    recordset.pop('links', None)
    return recordset