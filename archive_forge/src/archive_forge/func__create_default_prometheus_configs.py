import asyncio
import aiohttp
import logging
import os
import shutil
from typing import Optional
import psutil
from urllib.parse import quote
from ray.dashboard.modules.metrics.grafana_dashboard_factory import (
from ray.dashboard.modules.metrics.grafana_datasource_template import (
from ray.dashboard.modules.metrics.grafana_dashboard_provisioning_template import (
import ray.dashboard.optional_utils as dashboard_optional_utils
import ray.dashboard.utils as dashboard_utils
from ray.dashboard.consts import AVAILABLE_COMPONENT_NAMES_FOR_METRICS
def _create_default_prometheus_configs(self):
    """
        Creates the prometheus configurations that are by default provided by Ray.
        """
    prometheus_config_output_path = os.path.join(self._metrics_root, 'prometheus', 'prometheus.yml')
    if os.path.exists(prometheus_config_output_path):
        os.remove(prometheus_config_output_path)
    os.makedirs(os.path.dirname(prometheus_config_output_path), exist_ok=True)
    shutil.copy(PROMETHEUS_CONFIG_INPUT_PATH, prometheus_config_output_path)