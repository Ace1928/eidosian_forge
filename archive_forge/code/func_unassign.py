from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import CreatePrimaryIPResponse, PrimaryIP
def unassign(self, primary_ip: PrimaryIP | BoundPrimaryIP) -> BoundAction:
    """Unassigns a Primary IP, resulting in it being unreachable. You may assign it to a server again at a later time.

        :param primary_ip: :class:`BoundPrimaryIP <hcloud.primary_ips.client.BoundPrimaryIP>` or  :class:`PrimaryIP <hcloud.primary_ips.domain.PrimaryIP>`
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
    response = self._client.request(url=f'/primary_ips/{primary_ip.id}/actions/unassign', method='POST')
    return BoundAction(self._client.actions, response['action'])