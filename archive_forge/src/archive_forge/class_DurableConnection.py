import re
from typing import Dict, List
from xml.etree import ElementTree as ET  # noqa
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
class DurableConnection(ConnectionUserAndKey):
    host = API_HOST
    responseCls = DurableResponse

    def add_default_params(self, params):
        params['user_id'] = self.user_id
        params['key'] = self.key
        return params

    def add_default_headers(self, headers):
        headers['Content-Type'] = 'text/xml'
        headers['Content-Encoding'] = 'gzip; charset=ISO-8859-1'
        return headers