import ssl
import json
import time
import atexit
import base64
import asyncio
import hashlib
import logging
import warnings
import functools
import itertools
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
def _to_node_recursive(self, virtual_machine):
    summary = virtual_machine.summary
    name = summary.config.name
    path = summary.config.vmPathName
    memory = summary.config.memorySizeMB
    cpus = summary.config.numCpu
    disk = ''
    if summary.storage.committed:
        disk = summary.storage.committed / 1024 ** 3
    id_to_hash = str(memory) + str(cpus) + str(disk)
    size_id = hashlib.md5(id_to_hash.encode('utf-8')).hexdigest()
    size_name = name + '-size'
    size_extra = {'cpus': cpus}
    driver = self
    size = NodeSize(id=size_id, name=size_name, ram=memory, disk=disk, bandwidth=0, price=0, driver=driver, extra=size_extra)
    operating_system = summary.config.guestFullName
    host = summary.runtime.host
    os_type = 'unix'
    if 'Microsoft' in str(operating_system):
        os_type = 'windows'
    uuid = summary.config.instanceUuid
    annotation = summary.config.annotation
    state = summary.runtime.powerState
    status = self.NODE_STATE_MAP.get(state, NodeState.UNKNOWN)
    boot_time = summary.runtime.bootTime
    ip_addresses = []
    if summary.guest is not None:
        ip_addresses.append(summary.guest.ipAddress)
    overall_status = str(summary.overallStatus)
    public_ips = []
    private_ips = []
    extra = {'path': path, 'operating_system': operating_system, 'os_type': os_type, 'memory_MB': memory, 'cpus': cpus, 'overallStatus': overall_status, 'metadata': {}, 'snapshots': []}
    if disk:
        extra['disk'] = disk
    if host:
        extra['host'] = host.name
        parent = host.parent
        while parent:
            if isinstance(parent, vim.ClusterComputeResource):
                extra['cluster'] = parent.name
                break
            parent = parent.parent
    if boot_time:
        extra['boot_time'] = boot_time.isoformat()
    if annotation:
        extra['annotation'] = annotation
    for ip_address in ip_addresses:
        try:
            if is_public_subnet(ip_address):
                public_ips.append(ip_address)
            else:
                private_ips.append(ip_address)
        except Exception:
            pass
    if virtual_machine.snapshot:
        snapshots = [{'id': s.id, 'name': s.name, 'description': s.description, 'created': s.createTime.strftime('%Y-%m-%d %H:%M'), 'state': s.state} for s in virtual_machine.snapshot.rootSnapshotList]
        extra['snapshots'] = snapshots
    for custom_field in virtual_machine.customValue:
        key_id = custom_field.key
        key = self.find_custom_field_key(key_id)
        extra['metadata'][key] = custom_field.value
    node = Node(id=uuid, name=name, state=status, size=size, public_ips=public_ips, private_ips=private_ips, driver=self, extra=extra)
    node._uuid = uuid
    return node