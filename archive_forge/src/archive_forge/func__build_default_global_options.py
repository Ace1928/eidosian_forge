import logging
import os
from collections import defaultdict
from concurrent.futures import as_completed, CancelledError, TimeoutError
from copy import deepcopy
from errno import EEXIST, ENOENT
from hashlib import md5
from io import StringIO
from os import environ, makedirs, stat, utime
from os.path import (
from posixpath import join as urljoin
from random import shuffle
from time import time
from threading import Thread
from queue import Queue
from queue import Empty as QueueEmpty
from urllib.parse import quote
import json
from swiftclient import Connection
from swiftclient.command_helpers import (
from swiftclient.utils import (
from swiftclient.exceptions import ClientException
from swiftclient.multithreading import MultiThreadingManager
def _build_default_global_options():
    return {'snet': False, 'verbose': 1, 'debug': False, 'info': False, 'auth': environ.get('ST_AUTH'), 'auth_version': environ.get('ST_AUTH_VERSION', '1.0'), 'user': environ.get('ST_USER'), 'key': environ.get('ST_KEY'), 'retries': 5, 'retry_on_ratelimit': True, 'force_auth_retry': False, 'os_username': environ.get('OS_USERNAME'), 'os_user_id': environ.get('OS_USER_ID'), 'os_user_domain_name': environ.get('OS_USER_DOMAIN_NAME'), 'os_user_domain_id': environ.get('OS_USER_DOMAIN_ID'), 'os_password': environ.get('OS_PASSWORD'), 'os_tenant_id': environ.get('OS_TENANT_ID'), 'os_tenant_name': environ.get('OS_TENANT_NAME'), 'os_project_name': environ.get('OS_PROJECT_NAME'), 'os_project_id': environ.get('OS_PROJECT_ID'), 'os_project_domain_name': environ.get('OS_PROJECT_DOMAIN_NAME'), 'os_project_domain_id': environ.get('OS_PROJECT_DOMAIN_ID'), 'os_auth_url': environ.get('OS_AUTH_URL'), 'os_auth_token': environ.get('OS_AUTH_TOKEN'), 'os_auth_type': environ.get('OS_AUTH_TYPE'), 'os_application_credential_id': environ.get('OS_APPLICATION_CREDENTIAL_ID'), 'os_application_credential_secret': environ.get('OS_APPLICATION_CREDENTIAL_SECRET'), 'os_storage_url': environ.get('OS_STORAGE_URL'), 'os_region_name': environ.get('OS_REGION_NAME'), 'os_service_type': environ.get('OS_SERVICE_TYPE'), 'os_endpoint_type': environ.get('OS_ENDPOINT_TYPE'), 'os_cacert': environ.get('OS_CACERT'), 'os_cert': environ.get('OS_CERT'), 'os_key': environ.get('OS_KEY'), 'insecure': config_true_value(environ.get('SWIFTCLIENT_INSECURE')), 'ssl_compression': False, 'segment_threads': 10, 'object_dd_threads': 10, 'object_uu_threads': 10, 'container_threads': 10}