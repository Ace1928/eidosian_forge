from __future__ import annotations
from typing import TYPE_CHECKING
from ..core import BaseDomain
class CreatePlacementGroupResponse(BaseDomain):
    """Create Placement Group Response Domain

    :param placement_group: :class:`BoundPlacementGroup <hcloud.placement_groups.client.BoundPlacementGroup>`
           The Placement Group which was created
    :param action: :class:`BoundAction <hcloud.actions.client.BoundAction>`
           The Action which shows the progress of the Placement Group Creation
    """
    __slots__ = ('placement_group', 'action')

    def __init__(self, placement_group: BoundPlacementGroup, action: BoundAction | None):
        self.placement_group = placement_group
        self.action = action