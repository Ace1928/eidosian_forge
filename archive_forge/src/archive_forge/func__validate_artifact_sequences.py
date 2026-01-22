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
def _validate_artifact_sequences(self, seqs: Iterable[ArtifactSequence], *, incremental: bool=True, download_files_and_compare: bool=False, check_entries_are_downloadable: bool=True, remapping: Optional[Dict[Namespace, Namespace]]=None):
    if incremental:
        logger.info('Validating in incremental mode')

        def filtered_sequences():
            for seq in seqs:
                if not seq.artifacts:
                    continue
                art = seq.artifacts[0]
                try:
                    logged_by = _get_run_or_dummy_from_art(art, self.src_api)
                except requests.HTTPError as e:
                    logger.error(f'Validate Artifact http error: art.entity={art.entity!r}, art.project={art.project!r}, art.name={art.name!r}, e={e!r}')
                    continue
                if art.type == 'wandb-history' and isinstance(logged_by, _DummyRun):
                    continue
                yield seq
        artifacts = self._filter_previously_checked_artifacts(filtered_sequences())
    else:
        logger.info('Validating in non-incremental mode')
        artifacts = [art for seq in seqs for art in seq.artifacts]

    def _validate_artifact_wrapped(args):
        art, entity, project = args
        if remapping is not None and (namespace := Namespace(entity, project)) in remapping:
            remapped_ns = remapping[namespace]
            entity = remapped_ns.entity
            project = remapped_ns.project
        logger.debug(f'Validating art={art!r}, entity={entity!r}, project={project!r}')
        result = self._validate_artifact(art, entity, project, download_files_and_compare=download_files_and_compare, check_entries_are_downloadable=check_entries_are_downloadable)
        logger.debug(f'Finished validating art={art!r}, entity={entity!r}, project={project!r}')
        return result
    args = ((art, art.entity, art.project) for art in artifacts)
    art_problems = for_each(_validate_artifact_wrapped, args)
    for art, dst_entity, dst_project, problems in art_problems:
        name, ver = _get_art_name_ver(art)
        d = {'src_entity': art.entity, 'src_project': art.project, 'dst_entity': dst_entity, 'dst_project': dst_project, 'name': name, 'version': ver, 'type': art.type}
        if problems:
            d['problems'] = problems
            fname = ARTIFACT_ERRORS_FNAME
        else:
            fname = ARTIFACT_SUCCESSES_FNAME
        with open(fname, 'a') as f:
            f.write(json.dumps(d) + '\n')