import json
import os
import tempfile
import time
import urllib
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional
from wandb_gql import gql
import wandb
from wandb import env, util
from wandb.apis import public
from wandb.apis.attrs import Attrs
from wandb.apis.internal import Api as InternalApi
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.paginator import Paginator
from wandb.apis.public.const import RETRY_TIMEDELTA
from wandb.sdk.lib import ipython, json_util, runid
from wandb.sdk.lib.paths import LogicalPath
def _full_history(self, samples=500, stream='default'):
    node = 'history' if stream == 'default' else 'events'
    query = gql('\n        query RunFullHistory($project: String!, $entity: String!, $name: String!, $samples: Int) {\n            project(name: $project, entityName: $entity) {\n                run(name: $name) { %s(samples: $samples) }\n            }\n        }\n        ' % node)
    response = self._exec(query, samples=samples)
    return [json.loads(line) for line in response['project']['run'][node]]