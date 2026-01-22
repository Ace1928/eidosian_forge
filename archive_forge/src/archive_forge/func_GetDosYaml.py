from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from xml.etree import ElementTree
import ipaddr
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
def GetDosYaml(unused_application, dos_xml_str):
    return _MakeDosListIntoYaml(DosXmlParser().ProcessXml(dos_xml_str))