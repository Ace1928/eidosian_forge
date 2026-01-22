from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class UrlMapPathmatchersArray(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = []

    def to_request(self):
        items = []
        for item in self.request:
            items.append(self._request_for_item(item))
        return items

    def from_response(self):
        items = []
        for item in self.request:
            items.append(self._response_from_item(item))
        return items

    def _request_for_item(self, item):
        return remove_nones_from_dict({u'defaultService': replace_resource_dict(item.get(u'default_service', {}), 'selfLink'), u'description': item.get('description'), u'headerAction': UrlMapHeaderaction(item.get('header_action', {}), self.module).to_request(), u'name': item.get('name'), u'pathRules': UrlMapPathrulesArray(item.get('path_rules', []), self.module).to_request(), u'routeRules': UrlMapRouterulesArray(item.get('route_rules', []), self.module).to_request(), u'defaultUrlRedirect': UrlMapDefaulturlredirect(item.get('default_url_redirect', {}), self.module).to_request(), u'defaultRouteAction': UrlMapDefaultrouteaction(item.get('default_route_action', {}), self.module).to_request()})

    def _response_from_item(self, item):
        return remove_nones_from_dict({u'defaultService': item.get(u'defaultService'), u'description': item.get(u'description'), u'headerAction': UrlMapHeaderaction(item.get(u'headerAction', {}), self.module).from_response(), u'name': item.get(u'name'), u'pathRules': UrlMapPathrulesArray(item.get(u'pathRules', []), self.module).from_response(), u'routeRules': UrlMapRouterulesArray(item.get(u'routeRules', []), self.module).from_response(), u'defaultUrlRedirect': UrlMapDefaulturlredirect(item.get(u'defaultUrlRedirect', {}), self.module).from_response(), u'defaultRouteAction': UrlMapDefaultrouteaction(item.get(u'defaultRouteAction', {}), self.module).from_response()})