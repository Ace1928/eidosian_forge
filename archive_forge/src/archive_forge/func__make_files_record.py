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
def _make_files_record(self, artifacts: bool, files: bool, media: bool, code: bool) -> pb.Record:
    run_files = self.run.files()
    metadata_fname = f'{self.run_dir}/files/wandb-metadata.json'
    if not files or run_files is None:
        metadata_fname = self._make_metadata_file()
        run_files = [(metadata_fname, 'end')]
    files_record = pb.FilesRecord()
    for path, policy in run_files:
        if not artifacts and path.startswith('artifact/'):
            continue
        if not media and path.startswith('media/'):
            continue
        if not code and path.startswith('code/'):
            continue
        if 'media' in path:
            p = Path(path)
            path = str(p.relative_to(f'{self.run_dir}/files'))
        f = files_record.files.add()
        f.path = path
        f.policy = file_policy_to_enum(policy)
    return self.interface._make_record(files=files_record)