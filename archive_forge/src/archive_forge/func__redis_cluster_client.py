from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Any, List, Optional, Pattern
from urllib.parse import urlparse
import numpy as np
def _redis_cluster_client(redis_url: str, **kwargs: Any) -> RedisType:
    from redis.cluster import RedisCluster
    return RedisCluster.from_url(redis_url, **kwargs)