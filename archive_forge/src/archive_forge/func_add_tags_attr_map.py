import functools
import ipaddress
from openstackclient.identity import common as identity_common
from osc_lib import exceptions as osc_exc
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.api import exceptions
from octaviaclient.osc.v2 import constants
def add_tags_attr_map(attr_map):
    tags_attr_map = {'tags': ('tags', list), 'any_tags': ('tags-any', list), 'not_tags': ('not-tags', list), 'not_any_tags': ('not-tags-any', list)}
    attr_map.update(tags_attr_map)