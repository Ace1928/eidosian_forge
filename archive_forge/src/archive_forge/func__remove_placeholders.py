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
def _remove_placeholders(self, seq: ArtifactSequence) -> None:
    try:
        retry_arts_func = internal.exp_retry(self._dst_api.artifacts)
        dst_arts = list(retry_arts_func(seq.type_, seq.name))
    except wandb.CommError:
        logger.warn(f'seq={seq!r} does not exist in dst.  Has it already been deleted?')
        return
    except TypeError as e:
        logger.error(f'Problem getting dst versions (try again later) e={e!r}')
        return
    for art in dst_arts:
        if art.description != ART_SEQUENCE_DUMMY_PLACEHOLDER:
            continue
        if art.type in ('wandb-history', 'job'):
            continue
        try:
            art.delete(delete_aliases=True)
        except wandb.CommError as e:
            if 'cannot delete system managed artifact' in str(e):
                logger.warn('Cannot delete system managed artifact')
            else:
                raise e