from typing import Any
from typing import Dict
from typing import Optional
from ray._private import ray_option_utils
from ray.util.placement_group import PlacementGroup, check_placement_group_index
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
def validate_options(kwargs_dict: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if kwargs_dict is None:
        return None
    if len(kwargs_dict) == 0:
        return None
    out = {}
    for k, v in kwargs_dict.items():
        if k not in ray_option_utils.valid_options:
            raise ValueError(f"Invalid option keyword: '{k}'. {ray_option_utils.remote_args_error_string}")
        ray_option_utils.valid_options[k].validate(k, v)
        out[k] = v
    bundle_index = out.get('placement_group_bundle_index', None)
    pg = out.get('placement_group', None)
    scheduling_strategy = out.get('scheduling_strategy', None)
    if isinstance(scheduling_strategy, PlacementGroupSchedulingStrategy):
        pg = scheduling_strategy.placement_group
        bundle_index = scheduling_strategy.placement_group_bundle_index
    if bundle_index is not None:
        if pg is None:
            pg = PlacementGroup.empty()
        if pg == 'default' and out.get('placement_group_capture_child_tasks', None) is None:
            pg = PlacementGroup.empty()
        if isinstance(pg, PlacementGroup):
            check_placement_group_index(pg, bundle_index)
    return out