from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
Creates a RimuHosting instance

        @inherits: :class:`NodeDriver.create_node`

        :keyword    name: Must be a FQDN. e.g example.com.
        :type       name: ``str``

        :keyword    ex_billing_oid: If not set,
                                    a billing method is automatically picked.
        :type       ex_billing_oid: ``str``

        :keyword    ex_host_server_oid: The host server to set the VPS up on.
        :type       ex_host_server_oid: ``str``

        :keyword    ex_vps_order_oid_to_clone: Clone another VPS to use as
                                                the image for the new VPS.
        :type       ex_vps_order_oid_to_clone: ``str``

        :keyword    ex_num_ips: Number of IPs to allocate. Defaults to 1.
        :type       ex_num_ips: ``int``

        :keyword    ex_extra_ip_reason: Reason for needing the extra IPs.
        :type       ex_extra_ip_reason: ``str``

        :keyword    ex_memory_mb: Memory to allocate to the VPS.
        :type       ex_memory_mb: ``int``

        :keyword    ex_disk_space_mb: Diskspace to allocate to the VPS.
            Defaults to 4096 (4GB).
        :type       ex_disk_space_mb: ``int``

        :keyword    ex_disk_space_2_mb: Secondary disk size allocation.
                                        Disabled by default.
        :type       ex_disk_space_2_mb: ``int``

        :keyword    ex_control_panel: Control panel to install on the VPS.
        :type       ex_control_panel: ``str``
        