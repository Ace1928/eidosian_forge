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
def _build_xmltree(self):
    self.root = self._make_instantiation_root()
    if self.network is not None:
        instantiation_params = ET.SubElement(self.root, 'InstantiationParams')
        network_config_section = ET.SubElement(instantiation_params, 'NetworkConfigSection')
        ET.SubElement(network_config_section, 'Info', {'xmlns': 'http://schemas.dmtf.org/ovf/envelope/1'})
        network_config = ET.SubElement(network_config_section, 'NetworkConfig')
        self._add_network_association(network_config)
    if self.description is not None:
        ET.SubElement(self.root, 'Description').text = self.description
    self._add_vapp_template(self.root)