from http import client as http_client
import io
from unittest import mock
from oslo_serialization import jsonutils
import socket
from magnumclient.common.apiclient.exceptions import GatewayTimeout
from magnumclient.common.apiclient.exceptions import MultipleChoices
from magnumclient.common import httpclient as http
from magnumclient import exceptions as exc
from magnumclient.tests import utils
def _get_error_body(faultstring=None, debuginfo=None, err_type=NORMAL_ERROR):
    if err_type == NORMAL_ERROR:
        error_body = {'faultstring': faultstring, 'debuginfo': debuginfo}
        raw_error_body = jsonutils.dumps(error_body)
        body = {'error_message': raw_error_body}
    elif err_type == ERROR_DICT:
        body = {'error': {'title': faultstring, 'message': debuginfo}}
    elif err_type == ERROR_LIST_WITH_DETAIL:
        main_body = {'title': faultstring, 'detail': debuginfo}
        body = {'errors': [main_body]}
    elif err_type == ERROR_LIST_WITH_DESC:
        main_body = {'title': faultstring, 'description': debuginfo}
        body = {'errors': [main_body]}
    raw_body = jsonutils.dumps(body)
    return raw_body