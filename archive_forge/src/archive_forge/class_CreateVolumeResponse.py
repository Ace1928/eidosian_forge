from __future__ import annotations
from typing import TYPE_CHECKING
from ..core import BaseDomain, DomainIdentityMixin
class CreateVolumeResponse(BaseDomain):
    """Create Volume Response Domain

    :param volume: :class:`BoundVolume <hcloud.volumes.client.BoundVolume>`
           The created volume
    :param action: :class:`BoundAction <hcloud.actions.client.BoundAction>`
           The action that shows the progress of the Volume Creation
    :param next_actions: List[:class:`BoundAction <hcloud.actions.client.BoundAction>`]
           List of actions that are performed after the creation, like attaching to a server
    """
    __slots__ = ('volume', 'action', 'next_actions')

    def __init__(self, volume: BoundVolume, action: BoundAction, next_actions: list[BoundAction]):
        self.volume = volume
        self.action = action
        self.next_actions = next_actions