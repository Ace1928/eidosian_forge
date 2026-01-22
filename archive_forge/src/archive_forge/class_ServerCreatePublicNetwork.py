from __future__ import annotations
from typing import TYPE_CHECKING, Literal
from ..core import BaseDomain
class ServerCreatePublicNetwork(BaseDomain):
    """Server Create Public Network Domain

    :param ipv4: Optional[:class:`PrimaryIP <hcloud.primary_ips.domain.PrimaryIP>`]
    :param ipv6: Optional[:class:`PrimaryIP <hcloud.primary_ips.domain.PrimaryIP>`]
    :param enable_ipv4: bool
    :param enable_ipv6: bool
    """
    __slots__ = ('ipv4', 'ipv6', 'enable_ipv4', 'enable_ipv6')

    def __init__(self, ipv4: PrimaryIP | None=None, ipv6: PrimaryIP | None=None, enable_ipv4: bool=True, enable_ipv6: bool=True):
        self.ipv4 = ipv4
        self.ipv6 = ipv6
        self.enable_ipv4 = enable_ipv4
        self.enable_ipv6 = enable_ipv6