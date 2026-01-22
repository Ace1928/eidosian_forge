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
def _change_vm_names(self, vapp_or_vm_id, vm_names):
    if vm_names is None:
        return
    vms = self._get_vm_elements(vapp_or_vm_id)
    for i, vm in enumerate(vms):
        if len(vm_names) <= i:
            return
        res = self.connection.request('%s/guestCustomizationSection' % get_url_path(vm.get('href')))
        res.object.find(fixxpath(res.object, 'ComputerName')).text = vm_names[i]
        self._remove_admin_password(res.object)
        headers = {'Content-Type': 'application/vnd.vmware.vcloud.guestCustomizationSection+xml'}
        res = self.connection.request('%s/guestCustomizationSection' % get_url_path(vm.get('href')), data=ET.tostring(res.object), method='PUT', headers=headers)
        self._wait_for_task_completion(res.object.get('href'))
        req_xml = ET.Element('Vm', {'name': vm_names[i], 'xmlns': 'http://www.vmware.com/vcloud/v1.5'})
        res = self.connection.request(get_url_path(vm.get('href')), data=ET.tostring(req_xml), method='PUT', headers={'Content-Type': 'application/vnd.vmware.vcloud.vm+xml'})
        self._wait_for_task_completion(res.object.get('href'))