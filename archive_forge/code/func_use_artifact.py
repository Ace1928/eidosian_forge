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
@normalize_exceptions
def use_artifact(self, artifact, use_as=None):
    """Declare an artifact as an input to a run.

        Arguments:
            artifact (`Artifact`): An artifact returned from
                `wandb.Api().artifact(name)`
            use_as (string, optional): A string identifying
                how the artifact is used in the script. Used
                to easily differentiate artifacts used in a
                run, when using the beta wandb launch
                feature's artifact swapping functionality.

        Returns:
            A `Artifact` object.
        """
    api = InternalApi(default_settings={'entity': self.entity, 'project': self.project}, retry_timedelta=RETRY_TIMEDELTA)
    api.set_current_run_id(self.id)
    if isinstance(artifact, wandb.Artifact) and (not artifact.is_draft()):
        api.use_artifact(artifact.id, use_as=use_as or artifact.name)
        return artifact
    elif isinstance(artifact, wandb.Artifact) and artifact.is_draft():
        raise ValueError('Only existing artifacts are accepted by this api. Manually create one with `wandb artifact put`')
    else:
        raise ValueError('You must pass a wandb.Api().artifact() to use_artifact')