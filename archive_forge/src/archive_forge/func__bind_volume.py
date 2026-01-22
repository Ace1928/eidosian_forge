import json
import time
import hashlib
from datetime import datetime
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.kubernetes import (
def _bind_volume(self, volume, namespace='default'):
    """
        This method is for unbound volumes that were statically made.
        It will bind them to a pvc so they can be used by
        a kubernetes resource.
        """
    if volume.extra['is_bound']:
        return
    storage_class = volume.extra['storage_class_name']
    size = volume.size
    name = volume.name + '-pvc'
    volume_mode = volume.extra['volume_mode']
    access_mode = volume.extra['access_modes'][0]
    vol = self._create_volume_dynamic(size=size, name=name, storage_class_name=storage_class, volume_mode=volume_mode, namespace=namespace, access_mode=access_mode)
    return vol