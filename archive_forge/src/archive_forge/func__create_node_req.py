import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def _create_node_req(self, name, size, image, location, network=None, tags=None, metadata=None, boot_disk=None, external_ip='ephemeral', internal_ip=None, ex_disk_type='pd-standard', ex_disk_auto_delete=True, ex_service_accounts=None, description=None, ex_can_ip_forward=None, ex_disks_gce_struct=None, ex_nic_gce_struct=None, ex_on_host_maintenance=None, ex_automatic_restart=None, ex_preemptible=None, ex_subnetwork=None, ex_labels=None, ex_accelerator_type=None, ex_accelerator_count=None, ex_disk_size=None):
    """
        Returns a request and body to create a new node.

        This is a helper method to support both :class:`create_node` and
        :class:`ex_create_multiple_nodes`.

        :param  name: The name of the node to create.
        :type   name: ``str``

        :param  size: The machine type to use.
        :type   size: :class:`GCENodeSize`

        :param  image: The image to use to create the node (or, if using a
                       persistent disk, the image the disk was created from).
        :type   image: :class:`GCENodeImage` or ``None``

        :param  location: The location (zone) to create the node in.
        :type   location: :class:`NodeLocation` or :class:`GCEZone`

        :param  network: The network to associate with the node.
        :type   network: :class:`GCENetwork`

        :keyword  tags: A list of tags to associate with the node.
        :type     tags: ``list`` of ``str``

        :keyword  metadata: Metadata dictionary for instance.
        :type     metadata: ``dict``

        :keyword  boot_disk: Persistent boot disk to attach.
        :type     :class:`StorageVolume` or ``None``

        :keyword  external_ip: The external IP address to use.  If 'ephemeral'
                               (default), a new non-static address will be
                               used.  If 'None', then no external address will
                               be used.  To use an existing static IP address,
                               a GCEAddress object should be passed in. This
                               param will be ignored if also using the
                               ex_nic_gce_struct param.
        :type     external_ip: :class:`GCEAddress` or ``str`` or None

        :keyword  internal_ip: The private IP address to use.
        :type     internal_ip: :class:`GCEAddress` or ``str`` or ``None``

        :keyword  ex_disk_type: Specify a pd-standard (default) disk or pd-ssd
                                for an SSD disk.
        :type     ex_disk_type: ``str`` or :class:`GCEDiskType` or ``None``

        :keyword  ex_disk_auto_delete: Indicate that the boot disk should be
                                       deleted when the Node is deleted. Set to
                                       True by default.
        :type     ex_disk_auto_delete: ``bool``

        :keyword  ex_service_accounts: Specify a list of serviceAccounts when
                                       creating the instance. The format is a
                                       list of dictionaries containing email
                                       and list of scopes, e.g.
                                       [{'email':'default',
                                       'scopes':['compute', ...]}, ...]
                                       Scopes can either be full URLs or short
                                       names. If not provided, use the
                                       'default' service account email and a
                                       scope of 'devstorage.read_only'. Also
                                       accepts the aliases defined in
                                       'gcloud compute'.
        :type     ex_service_accounts: ``list``

        :keyword  description: The description of the node (instance).
        :type     description: ``str`` or ``None``

        :keyword  ex_can_ip_forward: Set to ``True`` to allow this node to
                                  send/receive non-matching src/dst packets.
        :type     ex_can_ip_forward: ``bool`` or ``None``

        :keyword  ex_disks_gce_struct: Support for passing in the GCE-specific
                                       formatted disks[] structure. No attempt
                                       is made to ensure proper formatting of
                                       the disks[] structure. Using this
                                       structure obviates the need of using
                                       other disk params like 'boot_disk',
                                       etc. See the GCE docs for specific
                                       details.
        :type     ex_disks_gce_struct: ``list`` or ``None``

        :keyword  ex_nic_gce_struct: Support passing in the GCE-specific
                                     formatted networkInterfaces[] structure.
                                     No attempt is made to ensure proper
                                     formatting of the networkInterfaces[]
                                     data. Using this structure obviates the
                                     need of using 'external_ip' and
                                     'ex_network'.  See the GCE docs for
                                     details.
        :type     ex_nic_gce_struct: ``list`` or ``None``

        :keyword  ex_on_host_maintenance: Defines whether node should be
                                          terminated or migrated when host
                                          machine goes down. Acceptable values
                                          are: 'MIGRATE' or 'TERMINATE' (If
                                          not supplied, value will be reset to
                                          GCE default value for the instance
                                          type.)
        :type     ex_on_host_maintenance: ``str`` or ``None``

        :keyword  ex_automatic_restart: Defines whether the instance should be
                                        automatically restarted when it is
                                        terminated by Compute Engine. (If not
                                        supplied, value will be set to the GCE
                                        default value for the instance type.)
        :type     ex_automatic_restart: ``bool`` or ``None``

        :keyword  ex_preemptible: Defines whether the instance is preemptible.
                                        (If not supplied, the instance will
                                         not be preemptible)
        :type     ex_preemptible: ``bool`` or ``None``

        :param  ex_subnetwork: The network to associate with the node.
        :type   ex_subnetwork: :class:`GCESubnetwork`

        :keyword  ex_disk_size: Specify the size of boot disk.
                                Integer in gigabytes.
        :type     ex_disk_size: ``int`` or ``None``

        :param  ex_labels: Label dict for node.
        :type   ex_labels: ``dict`` or ``None``

        :param  ex_accelerator_type: The accelerator to associate with the
                                     node.
        :type   ex_accelerator_type: :class:`GCEAcceleratorType` or ``None``

        :param  ex_accelerator_count: The number of accelerators to associate
                                      with the node.
        :type   ex_accelerator_count: ``int`` or ``None``

        :keyword  ex_disk_size: Specify size of the boot disk.
                                Integer in gigabytes.
        :type     ex_disk_size: ``int`` or ``None``

        :return:  A tuple containing a request string and a node_data dict.
        :rtype:   ``tuple`` of ``str`` and ``dict``
        """
    if not image and (not boot_disk) and (not ex_disks_gce_struct):
        raise ValueError("Missing root device or image. Must specify an 'image', existing 'boot_disk', or use the 'ex_disks_gce_struct'.")
    if boot_disk and ex_disks_gce_struct:
        raise ValueError("Cannot specify both 'boot_disk' and 'ex_disks_gce_struct'. Use one or the other.")
    use_selflinks = True
    source = None
    if boot_disk:
        source = boot_disk
    node_data = self._create_instance_properties(name, node_size=size, image=image, source=source, disk_type=ex_disk_type, disk_auto_delete=ex_disk_auto_delete, external_ip=external_ip, network=network, subnetwork=ex_subnetwork, can_ip_forward=ex_can_ip_forward, internal_ip=internal_ip, service_accounts=ex_service_accounts, on_host_maintenance=ex_on_host_maintenance, automatic_restart=ex_automatic_restart, preemptible=ex_preemptible, tags=tags, metadata=metadata, labels=ex_labels, description=description, disks_gce_struct=ex_disks_gce_struct, nic_gce_struct=ex_nic_gce_struct, accelerator_type=ex_accelerator_type, accelerator_count=ex_accelerator_count, use_selflinks=use_selflinks, disk_size=ex_disk_size)
    node_data['name'] = name
    request = '/zones/%s/instances' % location.name
    return (request, node_data)