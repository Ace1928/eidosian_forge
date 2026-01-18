from collections import abc
import xml.sax
import hashlib
import string
from boto.connection import AWSQueryConnection
from boto.exception import BotoServerError
import boto.mws.exception
import boto.mws.response
from boto.handler import XmlHandler
from boto.compat import filter, map, six, encodebytes
@requires(['ShipmentId'])
@structured_objects('InboundShipmentHeader', 'InboundShipmentItems')
@api_action('Inbound', 30, 0.5)
def update_inbound_shipment(self, request, response, **kw):
    """Updates an existing inbound shipment.  Amazon documentation
           is ambiguous as to whether the InboundShipmentHeader and
           InboundShipmentItems arguments are required.
        """
    return self._post_request(request, kw, response)