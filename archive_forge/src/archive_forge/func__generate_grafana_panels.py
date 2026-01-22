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
def _generate_grafana_panels(config: DashboardConfig, global_filters: List[str]) -> List[dict]:
    out = []
    panel_global_filters = [*config.standard_global_filters, *global_filters]
    for i, panel in enumerate(config.panels):
        template = copy.deepcopy(PANEL_TEMPLATE)
        template.update({'title': panel.title, 'description': panel.description, 'id': panel.id, 'targets': _generate_targets(panel, panel_global_filters)})
        if panel.grid_pos:
            template['gridPos'] = asdict(panel.grid_pos)
        else:
            template['gridPos']['y'] = i // 2
            template['gridPos']['x'] = 12 * (i % 2)
        template['yaxes'][0]['format'] = panel.unit
        template['fill'] = panel.fill
        template['stack'] = panel.stack
        template['linewidth'] = panel.linewidth
        out.append(template)
    return out