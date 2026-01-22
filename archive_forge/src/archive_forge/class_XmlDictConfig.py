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
class XmlDictConfig(dict):
    """
    Inherits from dict.  Looks for XML elements, such as attrib, that
    can be converted to a dictionary.  Any XML element that contains
    other XML elements, will be passed to XmlListConfig
    """

    def __init__(self, parent_element):
        if parent_element.items():
            if 'property' in parent_element.tag:
                self.update({parent_element.attrib.get('name'): parent_element.attrib.get('value')})
            else:
                self.update(dict(parent_element.items()))
        for element in parent_element:
            if len(element) > 0:
                if len(element) == 1 or element[0].tag != element[1].tag:
                    elem_dict = XmlDictConfig(element)
                else:
                    elem_dict = {element[0].tag.split('}')[1]: XmlListConfig(element)}
                if element.items():
                    elem_dict.update(dict(element.items()))
                self.update({element.tag.split('}')[1]: elem_dict})
            elif element.items():
                if element.tag.split('}')[1] in self:
                    if isinstance(self[element.tag.split('}')[1]], list):
                        self[element.tag.split('}')[1]].append(dict(element.items()))
                    else:
                        tmp_list = list()
                        tmp_dict = dict()
                        for k, v in self[element.tag.split('}')[1]].items():
                            if isinstance(k, XmlListConfig):
                                tmp_list.append(k)
                            else:
                                tmp_dict.update({k: v})
                        tmp_list.append(tmp_dict)
                        tmp_list.append(dict(element.items()))
                        self[element.tag.split('}')[1]] = tmp_list
                else:
                    self.update({element.tag.split('}')[1]: dict(element.items())})
            else:
                self.update({element.tag.split('}')[1]: element.text})