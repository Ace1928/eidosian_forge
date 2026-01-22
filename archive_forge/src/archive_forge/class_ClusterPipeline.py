import redis
from redis._parsers.helpers import bool_ok
from ..helpers import get_protocol_version, parse_to_list
from .commands import (
from .info import TSInfo
from .utils import parse_get, parse_m_get, parse_m_range, parse_range
class ClusterPipeline(TimeSeriesCommands, redis.cluster.ClusterPipeline):
    """Cluster pipeline for the module."""