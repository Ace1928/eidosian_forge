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
def _format_guest_accelerators(self, accelerator_type, accelerator_count):
    """
        Formats a GCE-friendly guestAccelerators request. Accepts an
        accelerator_type and accelerator_count that is wrapped up into a list
        of dictionaries for GCE to consume for a node creation request.

        :param  accelerator_type: Accelerator type to request.
        :type   accelerator_type: :class:`GCEAcceleratorType`

        :param  accelerator_count: Number of accelerators to request.
        :type   accelerator_count: ``int``

        :return: GCE-friendly guestAccelerators list of dictionaries.
        :rtype:  ``list``
        """
    accelerator_type = self._get_selflink_or_name(obj=accelerator_type, get_selflinks=True, objname='accelerator_type')
    return [{'acceleratorType': accelerator_type, 'acceleratorCount': accelerator_count}]