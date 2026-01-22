import re
from typing import Dict, List
from xml.etree import ElementTree as ET  # noqa
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
def _schema_builder(urn_nid, method, attributes):
    """
    Return a xml schema used to do an API request.

    :param urn_nid: API urn namespace id.
    :type urn_nid: type: ``str``

    :param method: API method.
    :type method: type: ``str``

    :param attributes: List of attributes to include.
    :type attributes: ``list`` of ``str``

    rtype: :class:`Element`
    """
    soap = ET.Element('soap:Body', {'xmlns:m': 'https://durabledns.com/services/dns/%s' % method})
    urn = ET.SubElement(soap, 'urn:{}:{}'.format(urn_nid, method))
    for attribute in attributes:
        ET.SubElement(urn, 'urn:{}:{}'.format(urn_nid, attribute))
    return soap