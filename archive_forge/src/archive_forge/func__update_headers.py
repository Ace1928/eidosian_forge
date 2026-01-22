import base64
import copy
import logging
import os
from oslo_utils import importutils
import requests
def _update_headers(self, headers):
    if not headers:
        headers = {}
    if isinstance(self.session, requests.Session):
        if self.auth_token:
            headers['X-Auth-Token'] = self.auth_token
        if self.project_id:
            headers['X-Project-Id'] = self.project_id
        if self.user_id:
            headers['X-User-Id'] = self.user_id
    if self.region_name:
        headers['X-Region-Name'] = self.region_name
    if self.target_auth_token:
        headers['X-Target-Auth-Token'] = self.target_auth_token
    if self.target_auth_uri:
        headers['X-Target-Auth-Uri'] = self.target_auth_uri
    if self.target_project_id:
        headers['X-Target-Project-Id'] = self.target_project_id
    if self.target_user_id:
        headers['X-Target-User-Id'] = self.target_user_id
    if self.target_insecure:
        headers['X-Target-Insecure'] = str(self.target_insecure)
    if self.target_region_name:
        headers['X-Target-Region-Name'] = self.target_region_name
    if self.target_user_domain_name:
        headers['X-Target-User-Domain-Name'] = self.target_user_domain_name
    if self.target_project_domain_name:
        h_name = 'X-Target-Project-Domain-Name'
        headers[h_name] = self.target_project_domain_name
    if self.target_service_catalog:
        headers['X-Target-Service-Catalog'] = base64.b64encode(self.target_service_catalog.encode('utf-8'))
    if osprofiler_web:
        headers.update(osprofiler_web.get_trace_id_headers())
    return headers