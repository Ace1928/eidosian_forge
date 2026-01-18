import ast
import asyncio
import base64
import datetime
import functools
import http.client
import json
import logging
import os
import re
import socket
import sys
import threading
from copy import deepcopy
from typing import (
import click
import requests
import yaml
from wandb_gql import Client, gql
from wandb_gql.client import RetryError
import wandb
from wandb import env, util
from wandb.apis.normalize import normalize_exceptions, parse_backend_error_messages
from wandb.errors import CommError, UnsupportedError, UsageError
from wandb.integration.sagemaker import parse_sm_secrets
from wandb.old.settings import Settings
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.lib.gql_request import GraphQLSession
from wandb.sdk.lib.hashutil import B64MD5, md5_file_b64
from ..lib import retry
from ..lib.filenames import DIFF_FNAME, METADATA_FNAME
from ..lib.gitlib import GitRepo
from . import context
from .progress import AsyncProgress, Progress
def upload_multipart_file_chunk(self, url: str, upload_chunk: bytes, extra_headers: Optional[Dict[str, str]]=None) -> Optional[requests.Response]:
    """Upload a file chunk to S3 with failure resumption.

        Arguments:
            url: The url to download
            upload_chunk: The path to the file you want to upload
            extra_headers: A dictionary of extra headers to send with the request

        Returns:
            The `requests` library response object
        """
    check_httpclient_logger_handler()
    try:
        if env.is_debug(env=self._environ):
            logger.debug('upload_file: %s', url)
        response = self._upload_file_session.put(url, data=upload_chunk, headers=extra_headers)
        if env.is_debug(env=self._environ):
            logger.debug('upload_file: %s complete', url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f'upload_file exception {url}: {e}')
        request_headers = e.request.headers if e.request is not None else ''
        logger.error(f'upload_file request headers: {request_headers}')
        response_content = e.response.content if e.response is not None else ''
        logger.error(f'upload_file response body: {response_content}')
        status_code = e.response.status_code if e.response is not None else 0
        is_aws_retryable = status_code == 400 and 'RequestTimeout' in str(response_content)
        if status_code in (308, 408, 409, 429, 500, 502, 503, 504) or isinstance(e, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)) or is_aws_retryable:
            _e = retry.TransientError(exc=e)
            raise _e.with_traceback(sys.exc_info()[2])
        else:
            wandb._sentry.reraise(e)
    return response