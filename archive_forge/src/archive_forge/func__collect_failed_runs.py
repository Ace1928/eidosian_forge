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
def _collect_failed_runs(self):
    if (df := _read_ndjson(RUN_ERRORS_FNAME)) is None:
        logger.debug(f'RUN_ERRORS_FNAME={RUN_ERRORS_FNAME!r} is empty, returning nothing')
        return
    unique_failed_runs = df[['src_entity', 'src_project', 'dst_entity', 'dst_project', 'run_id']].unique()
    for row in unique_failed_runs.iter_rows(named=True):
        src_entity = row['src_entity']
        src_project = row['src_project']
        run_id = row['run_id']
        run = self.src_api.run(f'{src_entity}/{src_project}/{run_id}')
        yield WandbRun(run, **self.run_api_kwargs)