from openstack.dns.v2 import floating_ip as _fip
from openstack.dns.v2 import recordset as _rs
from openstack.dns.v2 import zone as _zone
from openstack.dns.v2 import zone_export as _zone_export
from openstack.dns.v2 import zone_import as _zone_import
from openstack.dns.v2 import zone_share as _zone_share
from openstack.dns.v2 import zone_transfer as _zone_transfer
from openstack import proxy
def zone_imports(self, **query):
    """Retrieve a generator of zone imports

        :param dict query: Optional query parameters to be sent to limit the
            resources being returned.

            * `zone_id`: Zone I field.
            * `message`: Message field.
            * `status`: Status of the zone import record.

        :returns: A generator of zone
            :class:`~openstack.dns.v2.zone_import.ZoneImport` instances.
        """
    return self._list(_zone_import.ZoneImport, **query)