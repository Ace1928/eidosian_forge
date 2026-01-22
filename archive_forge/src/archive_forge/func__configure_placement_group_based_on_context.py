import warnings
from typing import Dict, List, Optional, Union
import ray
from ray._private.auto_init_hook import auto_init_ray
from ray._private.client_mode_hook import client_mode_should_convert, client_mode_wrap
from ray._private.utils import hex_to_binary, get_ray_doc_version
from ray._raylet import PlacementGroupID
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
def _configure_placement_group_based_on_context(placement_group_capture_child_tasks: bool, bundle_index: int, resources: Dict, placement_resources: Dict, task_or_actor_repr: str, placement_group: Union[PlacementGroup, str, None]='default') -> PlacementGroup:
    """Configure the placement group based on the given context.

    Based on the given context, this API returns the placement group instance
    for task/actor scheduling.

    Params:
        placement_group_capture_child_tasks: Whether or not the
            placement group needs to be captured from the global
            context.
        bundle_index: The bundle index for tasks/actor scheduling.
        resources: The scheduling resources.
        placement_resources: The scheduling placement resources for
            actors.
        task_or_actor_repr: The repr of task or actor
            function/class descriptor.
        placement_group: The placement group instance.
            - "default": Default placement group argument. Currently,
                the default behavior is to capture the parent task'
                placement group if placement_group_capture_child_tasks
                is set.
            - None: means placement group is explicitly not configured.
            - Placement group instance: In this case, do nothing.

    Returns:
        Placement group instance based on the given context.

    Raises:
        ValueError: If the bundle index is invalid for the placement group
            or the requested resources shape doesn't fit to any
            bundles.
    """
    assert placement_group_capture_child_tasks is not None
    assert resources is not None
    if placement_group != 'default':
        if not placement_group:
            placement_group = PlacementGroup.empty()
    elif placement_group == 'default':
        if placement_group_capture_child_tasks:
            placement_group = get_current_placement_group()
        else:
            placement_group = PlacementGroup.empty()
    if not placement_group:
        placement_group = PlacementGroup.empty()
    assert isinstance(placement_group, PlacementGroup)
    check_placement_group_index(placement_group, bundle_index)
    if not placement_group.is_empty:
        _validate_resource_shape(placement_group, resources, placement_resources, task_or_actor_repr)
    return placement_group