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
def _get_run_problems(self, src_run: Run, dst_run: Run, force_retry: bool=False) -> List[dict]:
    problems = []
    if force_retry:
        problems.append('__force_retry__')
    if (non_matching_metadata := self._compare_run_metadata(src_run, dst_run)):
        problems.append('metadata:' + str(non_matching_metadata))
    if (non_matching_summary := self._compare_run_summary(src_run, dst_run)):
        problems.append('summary:' + str(non_matching_summary))
    return problems