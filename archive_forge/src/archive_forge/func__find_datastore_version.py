import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
def _find_datastore_version(cs, datastore_version):
    """Get a datastore version by ID."""
    return utils.find_resource(cs.datastores, datastore_version)