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
def _sampled_history(self, keys, x_axis='_step', samples=500):
    spec = {'keys': [x_axis] + keys, 'samples': samples}
    query = gql('\n        query RunSampledHistory($project: String!, $entity: String!, $name: String!, $specs: [JSONString!]!) {\n            project(name: $project, entityName: $entity) {\n                run(name: $name) { sampledHistory(specs: $specs) }\n            }\n        }\n        ')
    response = self._exec(query, specs=[json.dumps(spec)])
    return response['project']['run']['sampledHistory'][0]