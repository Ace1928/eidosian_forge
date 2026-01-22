import datetime
import logging
from lxml import etree
import io
import warnings
import prov
import prov.identifier
from prov.model import DEFAULT_NAMESPACES, sorted_attributes
from prov.constants import *  # NOQA
from prov.serializers import Serializer
def _extract_attributes(element):
    """
    Extract the PROV attributes from an etree element.

    :param element: The lxml.etree.Element instance.
    """
    attributes = []
    for subel in element:
        sqname = etree.QName(subel)
        _t = xml_qname_to_QualifiedName(subel, '%s:%s' % (subel.prefix, sqname.localname))
        for key, value in subel.attrib.items():
            if key == _ns_xsi('type'):
                datatype = xml_qname_to_QualifiedName(subel, value)
                if datatype == XSD_QNAME:
                    _v = xml_qname_to_QualifiedName(subel, subel.text)
                else:
                    _v = prov.model.Literal(subel.text, datatype)
            elif key == _ns_prov('ref'):
                _v = xml_qname_to_QualifiedName(subel, value)
            elif key == _ns_xml('lang'):
                _v = prov.model.Literal(subel.text, langtag=value)
            else:
                warnings.warn("The element '%s' contains an attribute %s='%s' which is not representable in the prov module's internal data model and will thus be ignored." % (_t, str(key), str(value)), UserWarning)
        if not subel.attrib:
            _v = subel.text
        attributes.append((_t, _v))
    return attributes