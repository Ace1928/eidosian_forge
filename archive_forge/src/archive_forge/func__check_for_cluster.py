from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Any, List, Optional, Pattern
from urllib.parse import urlparse
import numpy as np
def _check_for_cluster(redis_client: RedisType) -> bool:
    import redis
    try:
        cluster_info = redis_client.info('cluster')
        return cluster_info['cluster_enabled'] == 1
    except redis.exceptions.RedisError:
        return False