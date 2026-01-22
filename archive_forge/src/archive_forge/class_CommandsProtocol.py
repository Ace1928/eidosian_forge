from datetime import datetime, timedelta
from typing import (
from redis.compat import Protocol
class CommandsProtocol(Protocol):
    connection_pool: Union['AsyncConnectionPool', 'ConnectionPool']

    def execute_command(self, *args, **options):
        ...