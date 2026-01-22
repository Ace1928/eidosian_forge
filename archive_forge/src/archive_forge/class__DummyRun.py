import itertools
import json
import logging
import numbers
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from unittest.mock import patch
import filelock
import polars as pl
import requests
import urllib3
import yaml
from wandb_gql import gql
import wandb
import wandb.apis.reports as wr
from wandb.apis.public import ArtifactCollection, Run
from wandb.apis.public.files import File
from wandb.apis.reports import Report
from wandb.util import coalesce, remove_keys_with_none_values
from . import validation
from .internals import internal
from .internals.protocols import PathStr, Policy
from .internals.util import Namespace, for_each
@dataclass
class _DummyRun:
    entity: str = ''
    project: str = ''
    run_id: str = RUN_DUMMY_PLACEHOLDER
    id: str = RUN_DUMMY_PLACEHOLDER
    display_name: str = RUN_DUMMY_PLACEHOLDER
    notes: str = ''
    url: str = ''
    group: str = ''
    created_at: str = '2000-01-01'
    user: _DummyUser = field(default_factory=_DummyUser)
    tags: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def files(self):
        return []