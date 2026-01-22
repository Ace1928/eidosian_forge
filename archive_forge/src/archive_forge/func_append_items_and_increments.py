from redis._parsers.helpers import bool_ok
from ..helpers import get_protocol_version, parse_to_list
from .commands import *  # noqa
from .info import BFInfo, CFInfo, CMSInfo, TDigestInfo, TopKInfo
@staticmethod
def append_items_and_increments(params, items, increments):
    """Append pairs of items and increments to params."""
    for i in range(len(items)):
        params.append(items[i])
        params.append(increments[i])