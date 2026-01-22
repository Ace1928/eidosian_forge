import re
import xml.etree.ElementTree as etree
from io import BytesIO
from copy import deepcopy
from time import sleep
from base64 import b64encode
from typing import Dict
from functools import wraps
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class XmlListConfig(list):
    """
    Creates a class from XML elements that make a list.  If a list of
    XML elements with attributes, the attributes are passed to XmlDictConfig.
    """

    def __init__(self, elem_list):
        for element in elem_list:
            if element is not None:
                if len(element) >= 0 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                elif element[0].tag == element[1].tag:
                    if 'property' in element.tag:
                        self.append({element.attrib.get('name'): element.attrib.get('value')})
                    else:
                        self.append(element.attrib)
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)