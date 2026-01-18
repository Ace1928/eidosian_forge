from openstack.dns.v2 import floating_ip as _fip
from openstack.dns.v2 import recordset as _rs
from openstack.dns.v2 import zone as _zone
from openstack.dns.v2 import zone_export as _zone_export
from openstack.dns.v2 import zone_import as _zone_import
from openstack.dns.v2 import zone_share as _zone_share
from openstack.dns.v2 import zone_transfer as _zone_transfer
from openstack import proxy
def update_zone_transfer_request(self, request, **attrs):
    """Update ZoneTransfer Request attributes

        :param floating_ip: The id or an instance of
            :class:`~openstack.dns.v2.zone_transfer.ZoneTransferRequest`.
        :param dict attrs: attributes for update on
            :class:`~openstack.dns.v2.zone_transfer.ZoneTransferRequest`.

        :rtype: :class:`~openstack.dns.v2.zone_transfer.ZoneTransferRequest`
        """
    return self._update(_zone_transfer.ZoneTransferRequest, request, **attrs)