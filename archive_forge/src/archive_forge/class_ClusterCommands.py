import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
class ClusterCommands(CommandsProtocol):
    """
    Class for Redis Cluster commands
    """

    def cluster(self, cluster_arg, *args, **kwargs) -> ResponseT:
        return self.execute_command(f'CLUSTER {cluster_arg.upper()}', *args, **kwargs)

    def readwrite(self, **kwargs) -> ResponseT:
        """
        Disables read queries for a connection to a Redis Cluster slave node.

        For more information see https://redis.io/commands/readwrite
        """
        return self.execute_command('READWRITE', **kwargs)

    def readonly(self, **kwargs) -> ResponseT:
        """
        Enables read queries for a connection to a Redis Cluster replica node.

        For more information see https://redis.io/commands/readonly
        """
        return self.execute_command('READONLY', **kwargs)