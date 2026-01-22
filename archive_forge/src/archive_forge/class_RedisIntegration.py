from __future__ import absolute_import
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk._compat import text_type
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
class RedisIntegration(Integration):
    identifier = 'redis'

    def __init__(self, max_data_size=_DEFAULT_MAX_DATA_SIZE):
        self.max_data_size = max_data_size

    @staticmethod
    def setup_once():
        try:
            from redis import StrictRedis, client
        except ImportError:
            raise DidNotEnable('Redis client not installed')
        _patch_redis(StrictRedis, client)
        _patch_redis_cluster()
        _patch_rb()
        try:
            _patch_rediscluster()
        except Exception:
            logger.exception('Error occurred while patching `rediscluster` library')