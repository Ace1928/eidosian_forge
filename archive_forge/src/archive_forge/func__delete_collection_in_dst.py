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
def _delete_collection_in_dst(self, seq: ArtifactSequence, namespace: Optional[Namespace]=None):
    """Deletes the equivalent artifact collection in destination.

        Intended to clear the destination when an uploaded artifact does not pass validation.
        """
    entity = coalesce(namespace.entity, seq.entity)
    project = coalesce(namespace.project, seq.project)
    art_type = f'{entity}/{project}/{seq.type_}'
    art_name = seq.name
    logger.info(f'Deleting collection entity={entity!r}, project={project!r}, art_type={art_type!r}, art_name={art_name!r}')
    try:
        dst_collection = self.dst_api.artifact_collection(art_type, art_name)
    except (wandb.CommError, ValueError):
        logger.warn(f"Collection doesn't exist art_type={art_type!r}, art_name={art_name!r}")
        return
    try:
        dst_collection.delete()
    except (wandb.CommError, ValueError) as e:
        logger.warn(f"Collection can't be deleted, art_type={art_type!r}, art_name={art_name!r}, e={e!r}")
        return