import datetime
import hashlib
import http.client as http
import os
import re
import urllib.parse as urlparse
import uuid
from castellan.common import exception as castellan_exception
from castellan import key_manager
import glance_store
from glance_store import location
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils as json
from oslo_utils import encodeutils
from oslo_utils import timeutils as oslo_timeutils
import requests
import webob.exc
from glance.api import common
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import store_utils
from glance.common import timeutils
from glance.common import utils
from glance.common import wsgi
from glance import context as glance_context
import glance.db
import glance.gateway
from glance.i18n import _, _LE, _LI, _LW
import glance.notifier
from glance.quota import keystone as ks_quota
import glance.schema
def _proxy_request_to_stage_host(self, image, req, body=None):
    """Proxy a request to a staging host.

        When an image was staged on another worker, that worker may record its
        worker_self_reference_url on the image, indicating that other workers
        should proxy requests to it while the image is staged. This method
        replays our current request against the remote host, returns the
        result, and performs any response error translation required.

        The remote request-id is used to replace the one on req.context so that
        a client sees the proper id used for the actual action.

        :param image: The Image from the repo
        :param req: The webob.Request from the current request
        :param body: The request body or None
        :returns: The result from the remote host
        :raises: webob.exc.HTTPClientError matching the remote's error, or
                 webob.exc.HTTPServerError if we were unable to contact the
                 remote host.
        """
    stage_host = image.extra_properties['os_glance_stage_host']
    LOG.info(_LI('Proxying %s request to host %s which has image staged'), req.method, stage_host)
    client = glance_context.get_ksa_client(req.context)
    url = '%s%s' % (stage_host, req.path)
    req_id_hdr = 'x-openstack-request-id'
    request_method = getattr(client, req.method.lower())
    try:
        r = request_method(url, json=body, timeout=60)
    except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout) as e:
        LOG.error(_LE('Failed to proxy to %r: %s'), url, e)
        raise webob.exc.HTTPGatewayTimeout('Stage host is unavailable')
    except requests.exceptions.RequestException as e:
        LOG.error(_LE('Failed to proxy to %r: %s'), url, e)
        raise webob.exc.HTTPBadGateway('Stage host is unavailable')
    req_id_hdr = 'x-openstack-request-id'
    if req_id_hdr in r.headers:
        LOG.debug('Replying with remote request id %s', r.headers[req_id_hdr])
        req.context.request_id = r.headers[req_id_hdr]
    if r.status_code // 100 != 2:
        raise proxy_response_error(r.status_code, r.reason)
    return image.image_id