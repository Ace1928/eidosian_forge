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
def _collect_failed_artifact_sequences(self) -> Iterable[ArtifactSequence]:
    if (df := _read_ndjson(ARTIFACT_ERRORS_FNAME)) is None:
        logger.debug(f'ARTIFACT_ERRORS_FNAME={ARTIFACT_ERRORS_FNAME!r} is empty, returning nothing')
        return
    unique_failed_sequences = df[['src_entity', 'src_project', 'name', 'type']].unique()
    for row in unique_failed_sequences.iter_rows(named=True):
        entity = row['src_entity']
        project = row['src_project']
        name = row['name']
        _type = row['type']
        art_name = f'{entity}/{project}/{name}'
        arts = self.src_api.artifacts(_type, art_name)
        arts = sorted(arts, key=lambda a: int(a.version.lstrip('v')))
        arts = sorted(arts, key=lambda a: a.type)
        yield ArtifactSequence(arts, entity, project, _type, name)