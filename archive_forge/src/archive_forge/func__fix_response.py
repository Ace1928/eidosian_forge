import re
from typing import Dict, List
from xml.etree import ElementTree as ET  # noqa
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
def _fix_response(self):
    items = re.findall('<ns1:.+ xmlns:ns1="">', self.body, flags=0)
    for item in items:
        parts = item.split(' ')
        prefix = parts[0].replace('<', '').split(':')[1]
        new_item = '<' + prefix + '>'
        close_tag = '</' + parts[0].replace('<', '') + '>'
        new_close_tag = '</' + prefix + '>'
        self.body = self.body.replace(item, new_item)
        self.body = self.body.replace(close_tag, new_close_tag)