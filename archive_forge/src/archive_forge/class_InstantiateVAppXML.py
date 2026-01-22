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
class InstantiateVAppXML:

    def __init__(self, name, template, net_href, cpus, memory, password=None, row=None, group=None):
        self.name = name
        self.template = template
        self.net_href = net_href
        self.cpus = cpus
        self.memory = memory
        self.password = password
        self.row = row
        self.group = group
        self._build_xmltree()

    def tostring(self):
        return ET.tostring(self.root)

    def _build_xmltree(self):
        self.root = self._make_instantiation_root()
        self._add_vapp_template(self.root)
        instantiation_params = ET.SubElement(self.root, 'InstantiationParams')
        self._make_product_section(instantiation_params)
        self._make_virtual_hardware(instantiation_params)
        network_config_section = ET.SubElement(instantiation_params, 'NetworkConfigSection')
        network_config = ET.SubElement(network_config_section, 'NetworkConfig')
        self._add_network_association(network_config)

    def _make_instantiation_root(self):
        return ET.Element('InstantiateVAppTemplateParams', {'name': self.name, 'xml:lang': 'en', 'xmlns': 'http://www.vmware.com/vcloud/v0.8', 'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance'})

    def _add_vapp_template(self, parent):
        return ET.SubElement(parent, 'VAppTemplate', {'href': self.template})

    def _make_product_section(self, parent):
        prod_section = ET.SubElement(parent, 'ProductSection', {'xmlns:q1': 'http://www.vmware.com/vcloud/v0.8', 'xmlns:ovf': 'http://schemas.dmtf.org/ovf/envelope/1'})
        if self.password:
            self._add_property(prod_section, 'password', self.password)
        if self.row:
            self._add_property(prod_section, 'row', self.row)
        if self.group:
            self._add_property(prod_section, 'group', self.group)
        return prod_section

    def _add_property(self, parent, ovfkey, ovfvalue):
        return ET.SubElement(parent, 'Property', {'xmlns': 'http://schemas.dmtf.org/ovf/envelope/1', 'ovf:key': ovfkey, 'ovf:value': ovfvalue})

    def _make_virtual_hardware(self, parent):
        vh = ET.SubElement(parent, 'VirtualHardwareSection', {'xmlns:q1': 'http://www.vmware.com/vcloud/v0.8'})
        self._add_cpu(vh)
        self._add_memory(vh)
        return vh

    def _add_cpu(self, parent):
        cpu_item = ET.SubElement(parent, 'Item', {'xmlns': 'http://schemas.dmtf.org/ovf/envelope/1'})
        self._add_instance_id(cpu_item, '1')
        self._add_resource_type(cpu_item, '3')
        self._add_virtual_quantity(cpu_item, self.cpus)
        return cpu_item

    def _add_memory(self, parent):
        mem_item = ET.SubElement(parent, 'Item', {'xmlns': 'http://schemas.dmtf.org/ovf/envelope/1'})
        self._add_instance_id(mem_item, '2')
        self._add_resource_type(mem_item, '4')
        self._add_virtual_quantity(mem_item, self.memory)
        return mem_item

    def _add_instance_id(self, parent, id):
        elm = ET.SubElement(parent, 'InstanceID', {'xmlns': 'http://schemas.dmtf.org/wbem/wscim/1/cim-schema/2/CIM_ResourceAllocationSettingData'})
        elm.text = id
        return elm

    def _add_resource_type(self, parent, type):
        elm = ET.SubElement(parent, 'ResourceType', {'xmlns': 'http://schemas.dmtf.org/wbem/wscim/1/cim-schema/2/CIM_ResourceAllocationSettingData'})
        elm.text = type
        return elm

    def _add_virtual_quantity(self, parent, amount):
        elm = ET.SubElement(parent, 'VirtualQuantity', {'xmlns': 'http://schemas.dmtf.org/wbem/wscim/1/cim-schema/2/CIM_ResourceAllocationSettingData'})
        elm.text = amount
        return elm

    def _add_network_association(self, parent):
        return ET.SubElement(parent, 'NetworkAssociation', {'href': self.net_href})