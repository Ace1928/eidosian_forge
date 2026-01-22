from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import (
def apply_to_resources(self, firewall: Firewall | BoundFirewall, resources: list[FirewallResource]) -> list[BoundAction]:
    """Applies one Firewall to multiple resources.

        :param firewall: :class:`BoundFirewall <hcloud.firewalls.client.BoundFirewall>` or  :class:`Firewall <hcloud.firewalls.domain.Firewall>`
        :param resources: List[:class:`FirewallResource <hcloud.firewalls.domain.FirewallResource>`]
        :return: List[:class:`BoundAction <hcloud.actions.client.BoundAction>`]
        """
    data: dict[str, Any] = {'apply_to': []}
    for resource in resources:
        data['apply_to'].append(resource.to_payload())
    response = self._client.request(url=f'/firewalls/{firewall.id}/actions/apply_to_resources', method='POST', json=data)
    return [BoundAction(self._client.actions, action_data) for action_data in response['actions']]