import os
import re
import copy
import time
import base64
import datetime
from xml.parsers.expat import ExpatError
from libcloud.utils.py3 import ET, b, next, httplib, urlparse, urlencode
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def _add_vm_disk(self, vapp_or_vm_id, vm_disk):
    if vm_disk is None:
        return
    rasd_ns = '{http://schemas.dmtf.org/wbem/wscim/1/cim-schema/2/CIM_ResourceAllocationSettingData}'
    vms = self._get_vm_elements(vapp_or_vm_id)
    for vm in vms:
        res = self.connection.request('%s/virtualHardwareSection/disks' % get_url_path(vm.get('href')))
        existing_ids = []
        new_disk = None
        for item in res.object.findall(fixxpath(res.object, 'Item')):
            for elem in item:
                if elem.tag == '%sInstanceID' % rasd_ns:
                    existing_ids.append(int(elem.text))
                if elem.tag in ['%sAddressOnParent' % rasd_ns, '%sParent' % rasd_ns]:
                    item.remove(elem)
            if item.find('%sHostResource' % rasd_ns) is not None:
                new_disk = item
        new_disk = copy.deepcopy(new_disk)
        disk_id = max(existing_ids) + 1
        new_disk.find('%sInstanceID' % rasd_ns).text = str(disk_id)
        new_disk.find('%sElementName' % rasd_ns).text = 'Hard Disk ' + str(disk_id)
        new_disk.find('%sHostResource' % rasd_ns).set(fixxpath(new_disk, 'capacity'), str(int(vm_disk) * 1024))
        res.object.append(new_disk)
        headers = {'Content-Type': 'application/vnd.vmware.vcloud.rasditemslist+xml'}
        res = self.connection.request('%s/virtualHardwareSection/disks' % get_url_path(vm.get('href')), data=ET.tostring(res.object), method='PUT', headers=headers)
        self._wait_for_task_completion(res.object.get('href'))