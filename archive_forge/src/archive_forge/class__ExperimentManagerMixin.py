from __future__ import annotations
import collections
import concurrent.futures as cf
import datetime
import functools
import itertools
import logging
import pathlib
import threading
import uuid
from contextvars import copy_context
from typing import (
from requests import HTTPError
from typing_extensions import TypedDict
import langsmith
from langsmith import env as ls_env
from langsmith import run_helpers as rh
from langsmith import run_trees, schemas
from langsmith import utils as ls_utils
from langsmith.evaluation.evaluator import (
from langsmith.evaluation.integrations import LangChainStringEvaluator
class _ExperimentManagerMixin:

    def __init__(self, /, experiment: Optional[Union[schemas.TracerSession, str]], metadata: Optional[dict]=None, client: Optional[langsmith.Client]=None):
        self.client = client or langsmith.Client()
        self._experiment: Optional[schemas.TracerSession] = None
        if experiment is None:
            self._experiment_name = _get_random_name()
        elif isinstance(experiment, str):
            self._experiment_name = experiment + '-' + str(uuid.uuid4().hex[:8])
        else:
            self._experiment_name = cast(str, experiment.name)
            self._experiment = experiment
        metadata = metadata or {}
        if not metadata.get('revision_id'):
            metadata = {'revision_id': ls_env.get_langchain_env_var_metadata().get('revision_id'), **metadata}
        self._metadata = metadata or {}

    @property
    def experiment_name(self) -> str:
        if self._experiment_name is not None:
            return self._experiment_name
        raise ValueError('Experiment name not provided, and experiment not yet started.')

    def _get_experiment(self) -> schemas.TracerSession:
        if self._experiment is None:
            raise ValueError('Experiment not started yet.')
        return self._experiment

    def _get_experiment_metadata(self):
        project_metadata = self._metadata or {}
        git_info = ls_env.get_git_info()
        if git_info:
            project_metadata = {**project_metadata, 'git': git_info}
        if self._experiment:
            project_metadata = {**self._experiment.metadata, **project_metadata}
        return project_metadata

    def _get_project(self, first_example: schemas.Example) -> schemas.TracerSession:
        if self._experiment is None:
            try:
                project_metadata = self._get_experiment_metadata()
                project = self.client.create_project(self.experiment_name, reference_dataset_id=first_example.dataset_id, metadata=project_metadata)
            except (HTTPError, ValueError, ls_utils.LangSmithError) as e:
                if 'already exists ' not in str(e):
                    raise e
                raise ValueError(f'Experiment {self.experiment_name} already exists. Please use a different name.')
        else:
            project = self._experiment
        return project

    def _print_experiment_start(self, project: schemas.TracerSession, first_example: schemas.Example) -> None:
        if project.url:
            project_url = project.url.split('?')[0]
            dataset_id = first_example.dataset_id
            base_url = project_url.split('/projects/p/')[0]
            comparison_url = f'{base_url}/datasets/{dataset_id}/compare?selectedSessions={project.id}'
            print(f"View the evaluation results for experiment: '{self.experiment_name}' at:\n{comparison_url}\n\n")
        else:
            print('Starting evaluation of experiment: %s', self.experiment_name)