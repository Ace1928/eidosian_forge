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
def _collect_reports(self, *, namespaces: Optional[Iterable[Namespace]]=None, limit: Optional[int]=None, api: Optional[Api]=None):
    api = coalesce(api, self.src_api)
    namespaces = coalesce(namespaces, self._all_namespaces())
    wandb.login(key=self.src_api_key, host=self.src_base_url)

    def reports():
        for ns in namespaces:
            for r in api.reports(ns.path):
                yield wr.Report.from_url(r.url, api=api)
    yield from itertools.islice(reports(), limit)