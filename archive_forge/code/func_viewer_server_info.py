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
@normalize_exceptions
def viewer_server_info(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    local_query = '\n            latestLocalVersionInfo {\n                outOfDate\n                latestVersionString\n                versionOnThisInstanceString\n            }\n        '
    cli_query = '\n            serverInfo {\n                cliVersionInfo\n                _LOCAL_QUERY_\n            }\n        '
    query_template = '\n        query Viewer{\n            viewer {\n                id\n                entity\n                username\n                email\n                flags\n                teams {\n                    edges {\n                        node {\n                            name\n                        }\n                    }\n                }\n            }\n            _CLI_QUERY_\n        }\n        '
    query_types, server_info_types, _ = self.server_info_introspection()
    cli_version_exists = 'serverInfo' in query_types and 'cliVersionInfo' in server_info_types
    local_version_exists = 'serverInfo' in query_types and 'latestLocalVersionInfo' in server_info_types
    cli_query_string = '' if not cli_version_exists else cli_query
    local_query_string = '' if not local_version_exists else local_query
    query_string = query_template.replace('_CLI_QUERY_', cli_query_string).replace('_LOCAL_QUERY_', local_query_string)
    query = gql(query_string)
    res = self.gql(query)
    return (res.get('viewer') or {}, res.get('serverInfo') or {})