import json
import time
import hashlib
from datetime import datetime
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.kubernetes import (
def _create_volume_dynamic(self, size, name, storage_class_name, volume_mode='Filesystem', namespace='default', access_mode='ReadWriteOnce'):
    """
        Method to create a Persistent Volume Claim for storage,
        thus storage is required in the arguments.
        This method assumes dynamic provisioning of the
        Persistent Volume so the storage_class given should
        allow for it (by default it usually is), or already
        have unbounded Persistent Volumes created by an admin.

        :param name: The name of the pvc an arbitrary string of lower letters
        :type name: `str`

        :param size: An int of the amount of gigabytes desired
        :type size: `int`

        :param namespace: The namespace where the claim will live
        :type namespace: `str`

        :param storage_class_name: If you want the pvc to be bound to
                                 a particular class of PVs specified here.
        :type storage_class_name: `str`

        :param access_mode: The desired access mode, ie "ReadOnlyMany"
        :type access_mode: `str`

        :param matchLabels: A dictionary with the labels, ie:
                            {'release': 'stable,}
        :type matchLabels: `dict` with keys `str` and values `str`
        """
    pvc = {'apiVersion': 'v1', 'kind': 'PersistentVolumeClaim', 'metadata': {'name': name}, 'spec': {'accessModes': [], 'volumeMode': volume_mode, 'resources': {'requests': {'storage': ''}}}}
    pvc['spec']['accessModes'].append(access_mode)
    if storage_class_name is not None:
        pvc['spec']['storageClassName'] = storage_class_name
    else:
        raise ValueError('The storage class name must be provided of astorage class which allows for dynamic provisioning')
    pvc['spec']['resources']['requests']['storage'] = str(size) + 'Gi'
    method = 'POST'
    req = ROOT_URL + 'namespaces/' + namespace + '/persistentvolumeclaims'
    data = json.dumps(pvc)
    try:
        result = self.connection.request(req, method=method, data=data)
    except Exception:
        raise
    if result.object['status']['phase'] != 'Bound':
        for _ in range(3):
            req = ROOT_URL + 'namespaces/' + namespace + '/persistentvolumeclaims/' + name
            try:
                result = self.connection.request(req).object
            except Exception:
                raise
            if result['status']['phase'] == 'Bound':
                break
            time.sleep(3)
    volumes = self.list_volumes()
    for volume in volumes:
        if volume.extra['pvc']['name'] == name:
            return volume