from __future__ import annotations
from typing import TYPE_CHECKING, Any
from ..core import BaseDomain
class CreateFirewallResponse(BaseDomain):
    """Create Firewall Response Domain

    :param firewall: :class:`BoundFirewall <hcloud.firewalls.client.BoundFirewall>`
           The Firewall which was created
    :param actions: List[:class:`BoundAction <hcloud.actions.client.BoundAction>`]
           The Action which shows the progress of the Firewall Creation
    """
    __slots__ = ('firewall', 'actions')

    def __init__(self, firewall: BoundFirewall, actions: list[BoundAction] | None):
        self.firewall = firewall
        self.actions = actions