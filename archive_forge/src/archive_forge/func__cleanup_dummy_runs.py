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
def _cleanup_dummy_runs(self, *, namespaces: Optional[Iterable[Namespace]]=None, api: Optional[Api]=None, remapping: Optional[Dict[Namespace, Namespace]]=None) -> None:
    api = coalesce(api, self.dst_api)
    namespaces = coalesce(namespaces, self._all_namespaces())
    for ns in namespaces:
        if remapping and ns in remapping:
            ns = remapping[ns]
        logger.debug(f'Cleaning up, ns={ns!r}')
        try:
            runs = list(api.runs(ns.path, filters={'displayName': RUN_DUMMY_PLACEHOLDER}))
        except ValueError as e:
            if 'Could not find project' in str(e):
                logger.error('Could not find project, does it exist?')
                continue
        for run in runs:
            logger.debug(f'Deleting dummy run={run!r}')
            run.delete(delete_artifacts=False)