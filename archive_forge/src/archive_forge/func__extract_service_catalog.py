import glob
import hashlib
import importlib.util
import itertools
import json
import logging
import os
import pkgutil
import re
import urllib
from urllib import parse as urlparse
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1.identity import base
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
import requests
from cinderclient._i18n import _
from cinderclient import api_versions
from cinderclient import exceptions
import cinderclient.extension
def _extract_service_catalog(self, url, resp, body, extract_token=True):
    """See what the auth service told us and process the response.

        We may get redirected to another site, fail or actually get
        back a service catalog with a token and our endpoints.
        """
    if resp.status_code == 200 or resp.status_code == 201:
        try:
            self.auth_url = url
            self.auth_ref = access.create(resp=resp, body=body)
            self.service_catalog = self.auth_ref.service_catalog
            if extract_token:
                self.auth_token = self.auth_ref.auth_token
            management_url = self.service_catalog.url_for(region_name=self.region_name, interface=self.endpoint_type, service_type=self.service_type, service_name=self.service_name)
            self.management_url = management_url.rstrip('/')
            return None
        except exceptions.AmbiguousEndpoints:
            print('Found more than one valid endpoint. Use a more restrictive filter')
            raise
        except ValueError:
            raise exceptions.AuthorizationFailure()
        except exceptions.EndpointNotFound:
            print('Could not find any suitable endpoint. Correct region?')
            raise
    elif resp.status_code == 305:
        return resp.headers['location']
    else:
        raise exceptions.from_response(resp, body)