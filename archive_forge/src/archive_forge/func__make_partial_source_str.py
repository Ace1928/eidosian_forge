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
@staticmethod
def _make_partial_source_str(source: Any, job_info: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    """Construct use_artifact.partial.source_info.sourc as str."""
    source_type = job_info.get('source_type', '').strip()
    if source_type == 'artifact':
        info_source = job_info.get('source', {})
        source.artifact.artifact = info_source.get('artifact', '')
        source.artifact.entrypoint.extend(info_source.get('entrypoint', []))
        source.artifact.notebook = info_source.get('notebook', False)
    elif source_type == 'repo':
        source.git.git_info.remote = metadata.get('git', {}).get('remote', '')
        source.git.git_info.commit = metadata.get('git', {}).get('commit', '')
        source.git.entrypoint.extend(metadata.get('entrypoint', []))
        source.git.notebook = metadata.get('notebook', False)
    elif source_type == 'image':
        source.image.image = metadata.get('docker', '')
    else:
        raise ValueError('Invalid source type')
    source_str: str = source.SerializeToString()
    return source_str