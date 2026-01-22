import argparse
import collections
import datetime
import getpass
import logging
import os
import pprint
import sys
import time
from oslo_utils import netutils
from oslo_utils import strutils
from oslo_utils import timeutils
import novaclient
from novaclient import api_versions
from novaclient import base
from novaclient import client
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import shell
from novaclient import utils
from novaclient.v2 import availability_zones
from novaclient.v2 import quotas
from novaclient.v2 import servers
def _expand_dict_attr(collection, attr):
    """Expand item attribute whose value is a dict.

    Take a collection of items where the named attribute is known to have a
    dictionary value and replace the named attribute with multiple attributes
    whose names are the keys of the dictionary namespaced with the original
    attribute name.
    """
    for item in collection:
        field = getattr(item, attr)
        delattr(item, attr)
        for subkey in field.keys():
            setattr(item, attr + ':' + subkey, field[subkey])
            item.set_info(attr + ':' + subkey, field[subkey])