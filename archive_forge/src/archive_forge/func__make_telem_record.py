import json
import logging
import math
import os
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import numpy as np
from google.protobuf.json_format import ParseDict
from tenacity import retry, stop_after_attempt, wait_random_exponential
from wandb import Artifact
from wandb.proto import wandb_internal_pb2 as pb
from wandb.proto import wandb_settings_pb2
from wandb.proto import wandb_telemetry_pb2 as telem_pb
from wandb.sdk.interface.interface import file_policy_to_enum
from wandb.sdk.interface.interface_queue import InterfaceQueue
from wandb.sdk.internal import context
from wandb.sdk.internal.sender import SendManager
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.util import coalesce, recursive_cast_dictlike_to_dict
from .protocols import ImporterRun
def _make_telem_record(self) -> pb.Record:
    telem = telem_pb.TelemetryRecord()
    feature = telem_pb.Feature()
    feature.importer_mlflow = True
    telem.feature.CopyFrom(feature)
    cli_version = self.run.cli_version()
    if cli_version:
        telem.cli_version = cli_version
    python_version = self.run.python_version()
    if python_version:
        telem.python_version = python_version
    return self.interface._make_record(telemetry=telem)