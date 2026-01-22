import collections
import configparser
import re
from oslo_log import log as logging
from oslo_serialization import jsonutils
from pycadf import cadftaxonomy as taxonomy
from pycadf import cadftype
from pycadf import credential
from pycadf import endpoint
from pycadf import eventfactory as factory
from pycadf import host
from pycadf import identifier
from pycadf import resource
from pycadf import tag
from urllib import parse as urlparse
def _get_service_info(self, endp):
    service = Service(type=self._MAP.service_endpoints.get(endp['type'], taxonomy.UNKNOWN), name=endp['name'], id=endp['endpoints'][0].get('id', endp['name']), admin_endp=endpoint.Endpoint(name='admin', url=endp['endpoints'][0].get('adminURL', taxonomy.UNKNOWN)), private_endp=endpoint.Endpoint(name='private', url=endp['endpoints'][0].get('internalURL', taxonomy.UNKNOWN)), public_endp=endpoint.Endpoint(name='public', url=endp['endpoints'][0].get('publicURL', taxonomy.UNKNOWN)))
    return service