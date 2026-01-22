import datetime
import decimal
import re
from typing import (
from xml.dom.minidom import Document, parseString
from xml.dom.minidom import Element as XmlElement
from xml.parsers.expat import ExpatError
from ._utils import StreamType, deprecate_no_replacement
from .errors import PdfReadError
from .generic import ContentStream, PdfObject
class XmpInformation(PdfObject):
    """
    An object that represents Adobe XMP metadata.
    Usually accessed by :py:attr:`xmp_metadata()<pypdf.PdfReader.xmp_metadata>`

    Raises:
      PdfReadError: if XML is invalid
    """

    def __init__(self, stream: ContentStream) -> None:
        self.stream = stream
        try:
            data = self.stream.get_data()
            doc_root: Document = parseString(data)
        except ExpatError as e:
            raise PdfReadError(f'XML in XmpInformation was invalid: {e}')
        self.rdf_root: XmlElement = doc_root.getElementsByTagNameNS(RDF_NAMESPACE, 'RDF')[0]
        self.cache: Dict[Any, Any] = {}

    def write_to_stream(self, stream: StreamType, encryption_key: Union[None, str, bytes]=None) -> None:
        if encryption_key is not None:
            deprecate_no_replacement('the encryption_key parameter of write_to_stream', '5.0.0')
        self.stream.write_to_stream(stream)

    def get_element(self, about_uri: str, namespace: str, name: str) -> Iterator[Any]:
        for desc in self.rdf_root.getElementsByTagNameNS(RDF_NAMESPACE, 'Description'):
            if desc.getAttributeNS(RDF_NAMESPACE, 'about') == about_uri:
                attr = desc.getAttributeNodeNS(namespace, name)
                if attr is not None:
                    yield attr
                yield from desc.getElementsByTagNameNS(namespace, name)

    def get_nodes_in_namespace(self, about_uri: str, namespace: str) -> Iterator[Any]:
        for desc in self.rdf_root.getElementsByTagNameNS(RDF_NAMESPACE, 'Description'):
            if desc.getAttributeNS(RDF_NAMESPACE, 'about') == about_uri:
                for i in range(desc.attributes.length):
                    attr = desc.attributes.item(i)
                    if attr.namespaceURI == namespace:
                        yield attr
                for child in desc.childNodes:
                    if child.namespaceURI == namespace:
                        yield child

    def _get_text(self, element: XmlElement) -> str:
        text = ''
        for child in element.childNodes:
            if child.nodeType == child.TEXT_NODE:
                text += child.data
        return text
    dc_contributor = property(_getter_bag(DC_NAMESPACE, 'contributor'))
    '\n    Contributors to the resource (other than the authors).\n\n    An unsorted array of names.\n    '
    dc_coverage = property(_getter_single(DC_NAMESPACE, 'coverage'))
    'Text describing the extent or scope of the resource.'
    dc_creator = property(_getter_seq(DC_NAMESPACE, 'creator'))
    'A sorted array of names of the authors of the resource, listed in order\n    of precedence.'
    dc_date = property(_getter_seq(DC_NAMESPACE, 'date', _converter_date))
    '\n    A sorted array of dates (datetime.datetime instances) of significance to\n    the resource.\n\n    The dates and times are in UTC.\n    '
    dc_description = property(_getter_langalt(DC_NAMESPACE, 'description'))
    'A language-keyed dictionary of textual descriptions of the content of the\n    resource.'
    dc_format = property(_getter_single(DC_NAMESPACE, 'format'))
    'The mime-type of the resource.'
    dc_identifier = property(_getter_single(DC_NAMESPACE, 'identifier'))
    'Unique identifier of the resource.'
    dc_language = property(_getter_bag(DC_NAMESPACE, 'language'))
    'An unordered array specifying the languages used in the resource.'
    dc_publisher = property(_getter_bag(DC_NAMESPACE, 'publisher'))
    'An unordered array of publisher names.'
    dc_relation = property(_getter_bag(DC_NAMESPACE, 'relation'))
    'An unordered array of text descriptions of relationships to other\n    documents.'
    dc_rights = property(_getter_langalt(DC_NAMESPACE, 'rights'))
    'A language-keyed dictionary of textual descriptions of the rights the\n    user has to this resource.'
    dc_source = property(_getter_single(DC_NAMESPACE, 'source'))
    'Unique identifier of the work from which this resource was derived.'
    dc_subject = property(_getter_bag(DC_NAMESPACE, 'subject'))
    'An unordered array of descriptive phrases or keywrods that specify the\n    topic of the content of the resource.'
    dc_title = property(_getter_langalt(DC_NAMESPACE, 'title'))
    'A language-keyed dictionary of the title of the resource.'
    dc_type = property(_getter_bag(DC_NAMESPACE, 'type'))
    'An unordered array of textual descriptions of the document type.'
    pdf_keywords = property(_getter_single(PDF_NAMESPACE, 'Keywords'))
    'An unformatted text string representing document keywords.'
    pdf_pdfversion = property(_getter_single(PDF_NAMESPACE, 'PDFVersion'))
    'The PDF file version, for example 1.0, 1.3.'
    pdf_producer = property(_getter_single(PDF_NAMESPACE, 'Producer'))
    'The name of the tool that created the PDF document.'
    xmp_create_date = property(_getter_single(XMP_NAMESPACE, 'CreateDate', _converter_date))
    '\n    The date and time the resource was originally created.\n\n    The date and time are returned as a UTC datetime.datetime object.\n    '
    xmp_modify_date = property(_getter_single(XMP_NAMESPACE, 'ModifyDate', _converter_date))
    '\n    The date and time the resource was last modified.\n\n    The date and time are returned as a UTC datetime.datetime object.\n    '
    xmp_metadata_date = property(_getter_single(XMP_NAMESPACE, 'MetadataDate', _converter_date))
    '\n    The date and time that any metadata for this resource was last changed.\n\n    The date and time are returned as a UTC datetime.datetime object.\n    '
    xmp_creator_tool = property(_getter_single(XMP_NAMESPACE, 'CreatorTool'))
    'The name of the first known tool used to create the resource.'
    xmpmm_document_id = property(_getter_single(XMPMM_NAMESPACE, 'DocumentID'))
    'The common identifier for all versions and renditions of this resource.'
    xmpmm_instance_id = property(_getter_single(XMPMM_NAMESPACE, 'InstanceID'))
    'An identifier for a specific incarnation of a document, updated each\n    time a file is saved.'

    @property
    def custom_properties(self) -> Dict[Any, Any]:
        """
        Retrieve custom metadata properties defined in the undocumented pdfx
        metadata schema.

        Returns:
            A dictionary of key/value items for custom metadata properties.
        """
        if not hasattr(self, '_custom_properties'):
            self._custom_properties = {}
            for node in self.get_nodes_in_namespace('', PDFX_NAMESPACE):
                key = node.localName
                while True:
                    idx = key.find('ↂ')
                    if idx == -1:
                        break
                    key = key[:idx] + chr(int(key[idx + 1:idx + 5], base=16)) + key[idx + 5:]
                if node.nodeType == node.ATTRIBUTE_NODE:
                    value = node.nodeValue
                else:
                    value = self._get_text(node)
                self._custom_properties[key] = value
        return self._custom_properties