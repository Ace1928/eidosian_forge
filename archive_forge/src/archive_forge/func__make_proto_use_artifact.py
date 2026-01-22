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
def _make_proto_use_artifact(self, use_artifact: pb.UseArtifactRecord, job_name: str, job_info: Dict[str, Any], metadata: Dict[str, Any]) -> pb.UseArtifactRecord:
    use_artifact.partial.job_name = job_name
    use_artifact.partial.source_info._version = job_info.get('_version', '')
    use_artifact.partial.source_info.source_type = job_info.get('source_type', '')
    use_artifact.partial.source_info.runtime = job_info.get('runtime', '')
    src_str = self._make_partial_source_str(source=use_artifact.partial.source_info.source, job_info=job_info, metadata=metadata)
    use_artifact.partial.source_info.source.ParseFromString(src_str)
    return use_artifact