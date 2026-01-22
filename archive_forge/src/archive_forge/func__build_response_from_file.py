import http.client as httplib
import io
import logging
import netaddr
from oslo_utils import timeutils
from oslo_utils import uuidutils
import requests
import suds
from suds import cache
from suds import client
from suds import plugin
import suds.sax.element as element
from suds import transport
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware import vim_util
def _build_response_from_file(self, request):
    file_path = request.url[7:]
    with open(file_path, 'rb') as f:
        file_content = f.read()
        buff = bytearray(file_content.decode(), 'utf-8')
        resp = Response(buff)
        return self.build_response(request, resp)