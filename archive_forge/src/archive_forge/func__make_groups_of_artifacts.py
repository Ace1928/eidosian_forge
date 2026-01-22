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
def _make_groups_of_artifacts(seq: ArtifactSequence, start: int=0):
    prev_ver = start - 1
    for art in seq:
        name, ver = _get_art_name_ver(art)
        if ver - prev_ver > 1:
            yield [_make_dummy_art(name, art.type, v) for v in range(prev_ver + 1, ver)]
        yield [art]
        prev_ver = ver