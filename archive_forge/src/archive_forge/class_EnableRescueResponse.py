from __future__ import annotations
from typing import TYPE_CHECKING, Literal
from ..core import BaseDomain
class EnableRescueResponse(BaseDomain):
    """Enable Rescue Response Domain

    :param action: :class:`BoundAction <hcloud.actions.client.BoundAction>`
           Shows the progress of the server enable rescue action
    :param root_password: str
           The root password of the server in the rescue mode
    """
    __slots__ = ('action', 'root_password')

    def __init__(self, action: BoundAction, root_password: str):
        self.action = action
        self.root_password = root_password