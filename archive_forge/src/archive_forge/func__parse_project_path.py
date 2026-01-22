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
def _parse_project_path(self, path):
    """Return project and entity for project specified by path."""
    project = self.settings['project'] or 'uncategorized'
    entity = self.settings['entity'] or self.default_entity
    if path is None:
        return (entity, project)
    parts = path.split('/', 1)
    if len(parts) == 1:
        return (entity, path)
    return parts