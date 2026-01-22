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
def _collect_runs(self, *, namespaces: Optional[Iterable[Namespace]]=None, limit: Optional[int]=None, skip_ids: Optional[List[str]]=None, start_date: Optional[str]=None, api: Optional[Api]=None) -> Iterable[WandbRun]:
    api = coalesce(api, self.src_api)
    namespaces = coalesce(namespaces, self._all_namespaces())
    filters: Dict[str, Any] = {}
    if skip_ids is not None:
        filters['name'] = {'$nin': skip_ids}
    if start_date is not None:
        filters['createdAt'] = {'$gte': start_date}

    def _runs():
        for ns in namespaces:
            logger.debug(f'Collecting runs from ns={ns!r}')
            for run in api.runs(ns.path, filters=filters):
                yield WandbRun(run, **self.run_api_kwargs)
    runs = itertools.islice(_runs(), limit)
    yield from runs