import logging
import os
import sys
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Iterable, NewType, Optional, Tuple, Union
from wandb.proto import wandb_internal_pb2 as pb
from wandb.proto import wandb_telemetry_pb2 as tpb
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.artifacts.artifact_manifest import ArtifactManifest
from wandb.sdk.artifacts.staging import get_staging_dir
from wandb.sdk.lib import json_util as json
from wandb.util import (
from ..data_types.utils import history_dict_to_json, val_to_json
from ..lib.mailbox import MailboxHandle
from . import summary_record as sr
from .message_future import MessageFuture
def _summary_encode(self, value: Any, path_from_root: str) -> dict:
    """Normalize, compress, and encode sub-objects for backend storage.

        value: Object to encode.
        path_from_root: `str` dot separated string from the top-level summary to the
            current `value`.

        Returns:
            A new tree of dict's with large objects replaced with dictionaries
            with "_type" entries that say which type the original data was.
        """
    if isinstance(value, dict):
        json_value = {}
        for key, value in value.items():
            json_value[key] = self._summary_encode(value, path_from_root + '.' + key)
        return json_value
    else:
        friendly_value, converted = json_friendly(val_to_json(self._run, path_from_root, value, namespace='summary'))
        json_value, compressed = maybe_compress_summary(friendly_value, get_h5_typename(value))
        if compressed:
            pass
        return json_value