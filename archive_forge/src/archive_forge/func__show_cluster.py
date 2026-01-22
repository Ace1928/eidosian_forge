import os
from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient import exceptions
from magnumclient.i18n import _
def _show_cluster(cluster):
    del cluster._info['links']
    utils.print_dict(cluster._info)