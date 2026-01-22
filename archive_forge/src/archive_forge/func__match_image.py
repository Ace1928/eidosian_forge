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
def _match_image(cs, wanted_properties):
    image_list = cs.glance.list()
    images_matched = []
    match = set(wanted_properties)
    for img in image_list:
        img_dict = {}
        for key, value in img.to_dict().items():
            try:
                set([key, value])
            except TypeError:
                pass
            else:
                img_dict[key] = value
        if match == match.intersection(set(img_dict.items())):
            images_matched.append(img)
    return images_matched