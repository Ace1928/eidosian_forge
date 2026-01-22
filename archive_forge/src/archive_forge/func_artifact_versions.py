import json
import logging
import os
import urllib
from typing import TYPE_CHECKING, Any, Dict, Optional
import requests
from wandb_gql import Client, gql
from wandb_gql.client import RetryError
import wandb
from wandb import env, util
from wandb.apis import public
from wandb.apis.internal import Api as InternalApi
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.public.const import RETRY_TIMEDELTA
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.launch.utils import LAUNCH_DEFAULT_PROJECT
from wandb.sdk.lib import retry, runid
from wandb.sdk.lib.gql_request import GraphQLSession
@normalize_exceptions
def artifact_versions(self, type_name, name, per_page=50):
    """Deprecated, use artifacts(type_name, name) instead."""
    wandb.termwarn('Api.artifact_versions(type_name, name) is deprecated, use Api.artifacts(type_name, name) instead.')
    return self.artifacts(type_name, name, per_page=per_page)