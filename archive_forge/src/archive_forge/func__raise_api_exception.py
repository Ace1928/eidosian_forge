import copy
import urllib
from oslo_serialization import jsonutils
from keystoneauth1 import exceptions
from mistralclient import utils
@staticmethod
def _raise_api_exception(resp):
    try:
        error_data = resp.headers.get('Server-Error-Message', None) or get_json(resp).get('faultstring')
    except ValueError:
        error_data = resp.content
    raise APIException(error_code=resp.status_code, error_message=error_data)