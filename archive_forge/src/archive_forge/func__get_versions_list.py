import flask
from flask import request
import http.client
from oslo_serialization import jsonutils
from keystone.common import json_home
import keystone.conf
from keystone.server import flask as ks_flask
def _get_versions_list(identity_url):
    versions = {}
    versions['v3'] = {'id': 'v3.14', 'status': 'stable', 'updated': '2020-04-07T00:00:00Z', 'links': [{'rel': 'self', 'href': identity_url}], 'media-types': [{'base': 'application/json', 'type': MEDIA_TYPE_JSON % 'v3'}]}
    return versions