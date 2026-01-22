from __future__ import annotations
from typing import TYPE_CHECKING
from ..core import BaseDomain, DomainIdentityMixin
class CreateManagedCertificateResponse(BaseDomain):
    """Create Managed Certificate Response Domain

    :param certificate: :class:`BoundCertificate <hcloud.certificate.client.BoundCertificate>`
           The created server
    :param action: :class:`BoundAction <hcloud.actions.client.BoundAction>`
           Shows the progress of the certificate creation
    """
    __slots__ = ('certificate', 'action')

    def __init__(self, certificate: BoundCertificate, action: BoundAction):
        self.certificate = certificate
        self.action = action