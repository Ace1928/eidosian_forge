import os
import sys
import time
import errno
import base64
import logging
import datetime
import urllib.parse
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from libcloud.utils.py3 import b, httplib, urlparse, urlencode
from libcloud.common.base import BaseDriver, JsonResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, ProviderError
from libcloud.utils.connection import get_response_object
class GoogleAuthType:
    """
    SA (Service Account),
    IA (Installed Application),
    GCE (Auth from a GCE instance with service account enabled)
    GCS_S3 (Cloud Storage S3 interoperability authentication)
    """
    SA = 'SA'
    IA = 'IA'
    GCE = 'GCE'
    GCS_S3 = 'GCS_S3'
    ALL_TYPES = [SA, IA, GCE, GCS_S3]
    OAUTH2_TYPES = [SA, IA, GCE]

    @classmethod
    def guess_type(cls, user_id):
        if cls._is_sa(user_id):
            return cls.SA
        elif cls._is_gcs_s3(user_id):
            return cls.GCS_S3
        elif cls._is_installed_application(user_id):
            return cls.IA
        elif cls._is_gce():
            return cls.GCE
        else:
            return cls.IA

    @classmethod
    def is_oauth2(cls, auth_type):
        return auth_type in cls.OAUTH2_TYPES

    @staticmethod
    def _is_installed_application(user_id):
        return user_id.endswith('apps.googleusercontent.com')

    @staticmethod
    def _is_gce():
        """
        Checks if we can access the GCE metadata server.
        Mocked in libcloud.test.common.google.GoogleTestCase.
        """
        http_code, http_reason, body = _get_gce_metadata(retry_failed=False)
        if http_code == httplib.OK and body:
            return True
        return False

    @staticmethod
    def _is_gcs_s3(user_id):
        """
        Checks S3 key format: alphanumeric chars starting with GOOG.
        """
        return user_id.startswith('GOOG')

    @staticmethod
    def _is_sa(user_id):
        return user_id.endswith('.gserviceaccount.com')