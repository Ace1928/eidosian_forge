from __future__ import annotations
from typing import TYPE_CHECKING
import tornado.web
from streamlit.runtime.stats import CacheStat, StatsManager
from streamlit.web.server.server_util import emit_endpoint_deprecation_notice
@staticmethod
def _stats_to_text(stats: list[CacheStat]) -> str:
    metric_type = '# TYPE cache_memory_bytes gauge'
    metric_unit = '# UNIT cache_memory_bytes bytes'
    metric_help = '# HELP Total memory consumed by a cache.'
    openmetrics_eof = '# EOF\n'
    result = [metric_type, metric_unit, metric_help]
    result.extend((stat.to_metric_str() for stat in stats))
    result.append(openmetrics_eof)
    return '\n'.join(result)