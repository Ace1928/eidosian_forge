from __future__ import annotations
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from pip._vendor.cachecontrol.cache import BaseCache
Redis uses connection pooling, no need to close the connection.