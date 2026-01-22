import copy
from dataclasses import asdict
import json
import os
from typing import List, Tuple
import ray
from ray.dashboard.modules.metrics.dashboards.common import DashboardConfig, Panel
from ray.dashboard.modules.metrics.dashboards.default_dashboard_panels import (
from ray.dashboard.modules.metrics.dashboards.serve_dashboard_panels import (
from ray.dashboard.modules.metrics.dashboards.serve_deployment_dashboard_panels import (
from ray.dashboard.modules.metrics.dashboards.data_dashboard_panels import (
def _generate_grafana_dashboard(dashboard_config: DashboardConfig) -> str:
    """
    Returns:
      Tuple with format dashboard_content, uid
    """
    uid, global_filters = _read_configs_for_dashboard(dashboard_config)
    panels = _generate_grafana_panels(dashboard_config, global_filters)
    base_file_name = dashboard_config.base_json_file_name
    base_json = json.load(open(os.path.join(os.path.dirname(__file__), 'dashboards', base_file_name)))
    base_json['panels'] = panels
    global_filters_str = ','.join(global_filters)
    variables = base_json.get('templating', {}).get('list', [])
    for variable in variables:
        if 'definition' not in variable:
            continue
        variable['definition'] = variable['definition'].format(global_filters=global_filters_str)
        variable['query']['query'] = variable['query']['query'].format(global_filters=global_filters_str)
    tags = base_json.get('tags', []) or []
    tags.append(f'rayVersion:{ray.__version__}')
    base_json['tags'] = tags
    base_json['uid'] = uid
    ray_meta = base_json.get('rayMeta', []) or []
    ray_meta.append('supportsGlobalFilterOverride')
    base_json['rayMeta'] = ray_meta
    return (json.dumps(base_json, indent=4), uid)