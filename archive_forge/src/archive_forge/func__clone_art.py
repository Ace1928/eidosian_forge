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
def _clone_art(art: Artifact, root: Optional[str]=None):
    if root is None:
        root = f'{SRC_ART_PATH}/{art.name}'
    if (path := _download_art(art, root=root)) is None:
        raise ValueError(f'Problem downloading art={art!r}')
    name, _ = art.name.split(':v')
    new_art = Artifact(name, ART_DUMMY_PLACEHOLDER_TYPE)
    new_art._type = art.type
    new_art._created_at = art.created_at
    new_art._aliases = art.aliases
    new_art._description = art.description
    with patch('click.echo'):
        new_art.add_dir(path)
    return new_art