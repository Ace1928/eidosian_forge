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
def _instantiate_node(self, name, image, network_elem, vdc, vm_network, vm_fence, instantiate_timeout, description=None):
    instantiate_xml = Instantiate_1_5_VAppXML(name=name, template=image.id, network=network_elem, vm_network=vm_network, vm_fence=vm_fence, description=description)
    headers = {'Content-Type': 'application/vnd.vmware.vcloud.instantiateVAppTemplateParams+xml'}
    res = self.connection.request('%s/action/instantiateVAppTemplate' % get_url_path(vdc.id), data=instantiate_xml.tostring(), method='POST', headers=headers)
    vapp_name = res.object.get('name')
    vapp_href = res.object.get('href')
    task_href = res.object.find(fixxpath(res.object, 'Tasks/Task')).get('href')
    self._wait_for_task_completion(task_href, instantiate_timeout)
    return (vapp_name, vapp_href)