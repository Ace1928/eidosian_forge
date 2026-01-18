import json
from libcloud.compute.base import (
from libcloud.common.gig_g8 import G8Connection
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError

        Create volume

        :param size: Size of the volume to create in GiB
        :type  size: ``int``

        :param name: Name of the volume
        :type  name: ``str``

        :param description: Description of the volume
        :type  description: ``str``

        :param disk_type: Type of the disk depending on the G8
                            D for datadisk is always available
        :type  disk_type: ``str``

        :rtype: class:`StorageVolume`
        