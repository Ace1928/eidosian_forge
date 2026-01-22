from redis._parsers.helpers import bool_ok
from ..helpers import get_protocol_version, parse_to_list
from .commands import *  # noqa
from .info import BFInfo, CFInfo, CMSInfo, TDigestInfo, TopKInfo
@staticmethod
def append_values_and_weights(params, items, weights):
    """Append pairs of items and weights to params."""
    for i in range(len(items)):
        params.append(items[i])
        params.append(weights[i])