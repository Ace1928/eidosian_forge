import ast
import asyncio
import base64
import datetime
import functools
import http.client
import json
import logging
import os
import re
import socket
import sys
import threading
from copy import deepcopy
from typing import (
import click
import requests
import yaml
from wandb_gql import Client, gql
from wandb_gql.client import RetryError
import wandb
from wandb import env, util
from wandb.apis.normalize import normalize_exceptions, parse_backend_error_messages
from wandb.errors import CommError, UnsupportedError, UsageError
from wandb.integration.sagemaker import parse_sm_secrets
from wandb.old.settings import Settings
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.lib.gql_request import GraphQLSession
from wandb.sdk.lib.hashutil import B64MD5, md5_file_b64
from ..lib import retry
from ..lib.filenames import DIFF_FNAME, METADATA_FNAME
from ..lib.gitlib import GitRepo
from . import context
from .progress import AsyncProgress, Progress
@normalize_exceptions
def upsert_sweep(self, config: dict, controller: Optional[str]=None, launch_scheduler: Optional[str]=None, scheduler: Optional[str]=None, obj_id: Optional[str]=None, project: Optional[str]=None, entity: Optional[str]=None, state: Optional[str]=None) -> Tuple[str, List[str]]:
    """Upsert a sweep object.

        Arguments:
            config (dict): sweep config (will be converted to yaml)
            controller (str): controller to use
            launch_scheduler (str): launch scheduler to use
            scheduler (str): scheduler to use
            obj_id (str): object id
            project (str): project to use
            entity (str): entity to use
            state (str): state
        """
    project_query = '\n            project {\n                id\n                name\n                entity {\n                    id\n                    name\n                }\n            }\n        '
    mutation_str = '\n        mutation UpsertSweep(\n            $id: ID,\n            $config: String,\n            $description: String,\n            $entityName: String,\n            $projectName: String,\n            $controller: JSONString,\n            $scheduler: JSONString,\n            $state: String\n        ) {\n            upsertSweep(input: {\n                id: $id,\n                config: $config,\n                description: $description,\n                entityName: $entityName,\n                projectName: $projectName,\n                controller: $controller,\n                scheduler: $scheduler,\n                state: $state\n            }) {\n                sweep {\n                    name\n                    _PROJECT_QUERY_\n                }\n                configValidationWarnings\n            }\n        }\n        '
    mutation_4 = gql(mutation_str.replace('$controller: JSONString,', '$controller: JSONString,$launchScheduler: JSONString,').replace('controller: $controller,', 'controller: $controller,launchScheduler: $launchScheduler,').replace('_PROJECT_QUERY_', project_query))
    mutation_3 = gql(mutation_str.replace('_PROJECT_QUERY_', project_query))
    mutation_2 = gql(mutation_str.replace('_PROJECT_QUERY_', project_query).replace('configValidationWarnings', ''))
    mutation_1 = gql(mutation_str.replace('_PROJECT_QUERY_', '').replace('configValidationWarnings', ''))
    mutations = [mutation_4, mutation_3, mutation_2, mutation_1]
    config = self._validate_config_and_fill_distribution(config)
    config_str = yaml.dump(json.loads(json.dumps(config)))
    err: Optional[Exception] = None
    for mutation in mutations:
        try:
            variables = {'id': obj_id, 'config': config_str, 'description': config.get('description'), 'entityName': entity or self.settings('entity'), 'projectName': project or self.settings('project'), 'controller': controller, 'launchScheduler': launch_scheduler, 'scheduler': scheduler}
            if state:
                variables['state'] = state
            response = self.gql(mutation, variable_values=variables, check_retry_fn=util.no_retry_4xx)
        except UsageError as e:
            raise e
        except Exception as e:
            err = e
            continue
        err = None
        break
    if err:
        raise err
    sweep: Dict[str, Dict[str, Dict]] = response['upsertSweep']['sweep']
    project_obj: Dict[str, Dict] = sweep.get('project', {})
    if project_obj:
        self.set_setting('project', project_obj['name'])
        entity_obj: dict = project_obj.get('entity', {})
        if entity_obj:
            self.set_setting('entity', entity_obj['name'])
    warnings = response['upsertSweep'].get('configValidationWarnings', [])
    return (response['upsertSweep']['sweep']['name'], warnings)