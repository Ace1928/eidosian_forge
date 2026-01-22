import asyncio
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Callable, List, Tuple, Optional
import aiohttp.web
from aiohttp.web import Response
from abc import ABC, abstractmethod
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
import ray.dashboard.optional_utils as dashboard_optional_utils
import ray.dashboard.utils as dashboard_utils
from ray.dashboard.consts import (
from ray.dashboard.datacenter import DataSource
from ray.dashboard.modules.log.log_manager import LogsManager
from ray.dashboard.optional_utils import rest_response
from ray.dashboard.state_aggregator import StateAPIManager
from ray.dashboard.utils import Change
from ray.util.state.common import (
from ray.util.state.exception import DataSourceUnavailable
from ray.util.state.state_manager import StateDataSourceClient
from ray.util.state.util import convert_string_to_type
def _options_from_req(self, req: aiohttp.web.Request) -> ListApiOptions:
    """Obtain `ListApiOptions` from the aiohttp request."""
    limit = int(req.query.get('limit') if req.query.get('limit') is not None else DEFAULT_LIMIT)
    if limit > RAY_MAX_LIMIT_FROM_API_SERVER:
        raise ValueError(f'Given limit {limit} exceeds the supported limit {RAY_MAX_LIMIT_FROM_API_SERVER}. Use a lower limit.')
    timeout = int(req.query.get('timeout', 30))
    filters = self._get_filters_from_req(req)
    detail = convert_string_to_type(req.query.get('detail', False), bool)
    exclude_driver = convert_string_to_type(req.query.get('exclude_driver', True), bool)
    return ListApiOptions(limit=limit, timeout=timeout, filters=filters, detail=detail, exclude_driver=exclude_driver)