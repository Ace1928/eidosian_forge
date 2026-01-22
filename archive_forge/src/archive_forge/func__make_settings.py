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
def _make_settings(root_dir: str, settings_override: Optional[Dict[str, Any]]=None) -> SettingsStatic:
    _settings_override = coalesce(settings_override, {})
    default_settings: Dict[str, Any] = {'files_dir': os.path.join(root_dir, 'files'), 'root_dir': root_dir, 'sync_file': os.path.join(root_dir, 'txlog.wandb'), 'resume': 'false', 'program': None, 'ignore_globs': [], 'disable_job_creation': True, '_start_time': 0, '_offline': None, '_sync': True, '_live_policy_rate_limit': 15, '_live_policy_wait_time': 600, '_async_upload_concurrency_limit': None, '_file_stream_timeout_seconds': 60}
    combined_settings = {**default_settings, **_settings_override}
    settings_message = wandb_settings_pb2.Settings()
    ParseDict(combined_settings, settings_message)
    return SettingsStatic(settings_message)