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
def _filter_previously_checked_artifacts(self, seqs: Iterable[ArtifactSequence]):
    if (df := _read_ndjson(ARTIFACT_SUCCESSES_FNAME)) is None:
        logger.info(f'ARTIFACT_SUCCESSES_FNAME={ARTIFACT_SUCCESSES_FNAME!r} is empty, yielding all artifact sequences')
        for seq in seqs:
            yield from seq.artifacts
        return
    for seq in seqs:
        for art in seq:
            try:
                logged_by = _get_run_or_dummy_from_art(art, self.src_api)
            except requests.HTTPError as e:
                logger.error(f'Failed to get run, skipping: art={art!r}, e={e!r}')
                continue
            if art.type == 'wandb-history' and isinstance(logged_by, _DummyRun):
                logger.debug(f'Skipping history artifact art={art!r}')
                continue
            entity = art.entity
            project = art.project
            _type = art.type
            name, ver = _get_art_name_ver(art)
            filtered_df = df.filter((df['src_entity'] == entity) & (df['src_project'] == project) & (df['name'] == name) & (df['version'] == ver) & (df['type'] == _type))
            if len(filtered_df) == 0:
                yield art