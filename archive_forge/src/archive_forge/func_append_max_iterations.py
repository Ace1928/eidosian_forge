from redis._parsers.helpers import bool_ok
from ..helpers import get_protocol_version, parse_to_list
from .commands import *  # noqa
from .info import BFInfo, CFInfo, CMSInfo, TDigestInfo, TopKInfo
@staticmethod
def append_max_iterations(params, max_iterations):
    """Append MAXITERATIONS to params."""
    if max_iterations is not None:
        params.extend(['MAXITERATIONS', max_iterations])