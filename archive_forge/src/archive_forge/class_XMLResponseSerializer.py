import datetime
from lxml import etree
from oslo_log import log as logging
from oslo_serialization import jsonutils
class XMLResponseSerializer(object):

    def object_to_element(self, obj, element):
        if isinstance(obj, list):
            for item in obj:
                subelement = etree.SubElement(element, 'member')
                self.object_to_element(item, subelement)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                subelement = etree.SubElement(element, key)
                if key in JSON_ONLY_KEYS:
                    if value:
                        try:
                            subelement.text = jsonutils.dumps(value)
                        except TypeError:
                            subelement.text = str(value)
                else:
                    self.object_to_element(value, subelement)
        else:
            element.text = str(obj)

    def to_xml(self, data):
        root = next(iter(data.keys()))
        eltree = etree.Element(root)
        self.object_to_element(data.get(root), eltree)
        response = etree.tostring(eltree)
        return response

    def default(self, response, result):
        response.content_type = 'application/xml'
        response.body = self.to_xml(result)