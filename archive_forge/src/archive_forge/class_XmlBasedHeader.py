from io import BytesIO
from xml.etree.ElementTree import Element, SubElement, tostring  # noqa
from xml.parsers.expat import ParserCreate
from .filebasedimages import FileBasedHeader
class XmlBasedHeader(FileBasedHeader, XmlSerializable):
    """Basic wrapper around FileBasedHeader and XmlSerializable."""