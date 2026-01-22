import xml.dom.minidom
class XmlFix:
    """Object to manage temporary patching of xml.dom.minidom."""

    def __init__(self):
        self.write_data = xml.dom.minidom._write_data
        self.writexml = xml.dom.minidom.Element.writexml
        xml.dom.minidom._write_data = _Replacement_write_data
        xml.dom.minidom.Element.writexml = _Replacement_writexml

    def Cleanup(self):
        if self.write_data:
            xml.dom.minidom._write_data = self.write_data
            xml.dom.minidom.Element.writexml = self.writexml
            self.write_data = None

    def __del__(self):
        self.Cleanup()