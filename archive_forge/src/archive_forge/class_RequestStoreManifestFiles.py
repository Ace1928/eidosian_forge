import concurrent.futures
import functools
import os
import queue
import shutil
import threading
from typing import TYPE_CHECKING, NamedTuple, Optional, Union, cast
from wandb.filesync import step_upload
from wandb.sdk.lib import filesystem, runid
from wandb.sdk.lib.paths import LogicalPath
class RequestStoreManifestFiles(NamedTuple):
    manifest: 'ArtifactManifest'
    artifact_id: str
    save_fn: 'SaveFn'
    save_fn_async: 'SaveFnAsync'