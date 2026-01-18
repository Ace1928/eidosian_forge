from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal
from ..core import BaseDomain
def to_payload(self) -> dict[str, Any]:
    """
        Generates the request payload from this domain object.
        """
    payload: dict[str, Any] = {'type': self.type}
    if self.use_private_ip is not None:
        payload['use_private_ip'] = self.use_private_ip
    if self.type == 'server':
        if self.server is None:
            raise ValueError(f'server is not defined in target {self!r}')
        payload['server'] = {'id': self.server.id}
    elif self.type == 'label_selector':
        if self.label_selector is None:
            raise ValueError(f'label_selector is not defined in target {self!r}')
        payload['label_selector'] = {'selector': self.label_selector.selector}
    elif self.type == 'ip':
        if self.ip is None:
            raise ValueError(f'ip is not defined in target {self!r}')
        payload['ip'] = {'ip': self.ip.ip}
    return payload