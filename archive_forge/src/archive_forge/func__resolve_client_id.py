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
def _resolve_client_id(self, client_id: str) -> Optional[str]:
    if client_id in self._client_id_mapping:
        return self._client_id_mapping[client_id]
    query = gql('\n            query ClientIDMapping($clientID: ID!) {\n                clientIDMapping(clientID: $clientID) {\n                    serverID\n                }\n            }\n        ')
    response = self.gql(query, variable_values={'clientID': client_id})
    server_id = None
    if response is not None:
        client_id_mapping = response.get('clientIDMapping')
        if client_id_mapping is not None:
            server_id = client_id_mapping.get('serverID')
            if server_id is not None:
                self._client_id_mapping[client_id] = server_id
    return server_id