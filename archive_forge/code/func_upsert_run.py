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
def upsert_run(self, id: Optional[str]=None, name: Optional[str]=None, project: Optional[str]=None, host: Optional[str]=None, group: Optional[str]=None, tags: Optional[List[str]]=None, config: Optional[dict]=None, description: Optional[str]=None, entity: Optional[str]=None, state: Optional[str]=None, display_name: Optional[str]=None, notes: Optional[str]=None, repo: Optional[str]=None, job_type: Optional[str]=None, program_path: Optional[str]=None, commit: Optional[str]=None, sweep_name: Optional[str]=None, summary_metrics: Optional[str]=None, num_retries: Optional[int]=None) -> Tuple[dict, bool, Optional[List]]:
    """Update a run.

        Arguments:
            id (str, optional): The existing run to update
            name (str, optional): The name of the run to create
            group (str, optional): Name of the group this run is a part of
            project (str, optional): The name of the project
            host (str, optional): The name of the host
            tags (list, optional): A list of tags to apply to the run
            config (dict, optional): The latest config params
            description (str, optional): A description of this project
            entity (str, optional): The entity to scope this project to.
            display_name (str, optional): The display name of this project
            notes (str, optional): Notes about this run
            repo (str, optional): Url of the program's repository.
            state (str, optional): State of the program.
            job_type (str, optional): Type of job, e.g 'train'.
            program_path (str, optional): Path to the program.
            commit (str, optional): The Git SHA to associate the run with
            sweep_name (str, optional): The name of the sweep this run is a part of
            summary_metrics (str, optional): The JSON summary metrics
            num_retries (int, optional): Number of retries
        """
    query_string = '\n        mutation UpsertBucket(\n            $id: String,\n            $name: String,\n            $project: String,\n            $entity: String,\n            $groupName: String,\n            $description: String,\n            $displayName: String,\n            $notes: String,\n            $commit: String,\n            $config: JSONString,\n            $host: String,\n            $debug: Boolean,\n            $program: String,\n            $repo: String,\n            $jobType: String,\n            $state: String,\n            $sweep: String,\n            $tags: [String!],\n            $summaryMetrics: JSONString,\n        ) {\n            upsertBucket(input: {\n                id: $id,\n                name: $name,\n                groupName: $groupName,\n                modelName: $project,\n                entityName: $entity,\n                description: $description,\n                displayName: $displayName,\n                notes: $notes,\n                config: $config,\n                commit: $commit,\n                host: $host,\n                debug: $debug,\n                jobProgram: $program,\n                jobRepo: $repo,\n                jobType: $jobType,\n                state: $state,\n                sweep: $sweep,\n                tags: $tags,\n                summaryMetrics: $summaryMetrics,\n            }) {\n                bucket {\n                    id\n                    name\n                    displayName\n                    description\n                    config\n                    sweepName\n                    project {\n                        id\n                        name\n                        entity {\n                            id\n                            name\n                        }\n                    }\n                }\n                inserted\n                _Server_Settings_\n            }\n        }\n        '
    self.server_settings_introspection()
    server_settings_string = '\n        serverSettings {\n                serverMessages{\n                    utfText\n                    plainText\n                    htmlText\n                    messageType\n                    messageLevel\n                }\n         }\n        ' if self._server_settings_type else ''
    query_string = query_string.replace('_Server_Settings_', server_settings_string)
    mutation = gql(query_string)
    config_str = json.dumps(config) if config else None
    if not description or description.isspace():
        description = None
    kwargs = {}
    if num_retries is not None:
        kwargs['num_retries'] = num_retries
    variable_values = {'id': id, 'entity': entity or self.settings('entity'), 'name': name, 'project': project or util.auto_project_name(program_path), 'groupName': group, 'tags': tags, 'description': description, 'config': config_str, 'commit': commit, 'displayName': display_name, 'notes': notes, 'host': None if self.settings().get('anonymous') == 'true' else host, 'debug': env.is_debug(env=self._environ), 'repo': repo, 'program': program_path, 'jobType': job_type, 'state': state, 'sweep': sweep_name, 'summaryMetrics': summary_metrics}
    check_retry_fn = util.make_check_retry_fn(check_fn=util.check_retry_conflict_or_gone, check_timedelta=datetime.timedelta(minutes=2), fallback_retry_fn=util.no_retry_auth)
    response = self.gql(mutation, variable_values=variable_values, check_retry_fn=check_retry_fn, **kwargs)
    run_obj: Dict[str, Dict[str, Dict[str, str]]] = response['upsertBucket']['bucket']
    project_obj: Dict[str, Dict[str, str]] = run_obj.get('project', {})
    if project_obj:
        self.set_setting('project', project_obj['name'])
        entity_obj = project_obj.get('entity', {})
        if entity_obj:
            self.set_setting('entity', entity_obj['name'])
    server_messages = None
    if self._server_settings_type:
        server_messages = response['upsertBucket'].get('serverSettings', {}).get('serverMessages', [])
    return (response['upsertBucket']['bucket'], response['upsertBucket']['inserted'], server_messages)