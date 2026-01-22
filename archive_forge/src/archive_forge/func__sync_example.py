from __future__ import annotations
import atexit
import concurrent.futures
import datetime
import functools
import inspect
import logging
import threading
import uuid
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar, overload
import orjson
from typing_extensions import TypedDict
from langsmith import client as ls_client
from langsmith import env as ls_env
from langsmith import run_helpers as rh
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
def _sync_example(self, example_id: uuid.UUID, inputs: dict, outputs: dict, metadata: dict) -> None:
    inputs_ = _serde_example_values(inputs)
    outputs_ = _serde_example_values(outputs)
    try:
        example = self.client.read_example(example_id=example_id)
        if inputs_ != example.inputs or outputs_ != example.outputs or str(example.dataset_id) != str(self.id):
            self.client.update_example(example_id=example.id, inputs=inputs_, outputs=outputs_, metadata=metadata, dataset_id=self.id)
    except ls_utils.LangSmithNotFoundError:
        example = self.client.create_example(example_id=example_id, inputs=inputs_, outputs=outputs_, dataset_id=self.id, metadata=metadata)
    if example.modified_at:
        self.update_version(example.modified_at)