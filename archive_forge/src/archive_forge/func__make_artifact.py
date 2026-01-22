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
def _make_artifact(self, artifact: 'Artifact') -> pb.ArtifactRecord:
    proto_artifact = pb.ArtifactRecord()
    proto_artifact.type = artifact.type
    proto_artifact.name = artifact.name
    proto_artifact.client_id = artifact._client_id
    proto_artifact.sequence_client_id = artifact._sequence_client_id
    proto_artifact.digest = artifact.digest
    if artifact.distributed_id:
        proto_artifact.distributed_id = artifact.distributed_id
    if artifact.description:
        proto_artifact.description = artifact.description
    if artifact.metadata:
        proto_artifact.metadata = json.dumps(json_friendly_val(artifact.metadata))
    if artifact._base_id:
        proto_artifact.base_id = artifact._base_id
    ttl_duration_input = artifact._ttl_duration_seconds_to_gql()
    if ttl_duration_input:
        proto_artifact.ttl_duration_seconds = ttl_duration_input
    proto_artifact.incremental_beta1 = artifact.incremental
    self._make_artifact_manifest(artifact.manifest, obj=proto_artifact.manifest)
    return proto_artifact