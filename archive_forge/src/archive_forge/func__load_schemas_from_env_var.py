import os
import jsonschema
import logging
from typing import List
import json
from ray._private.runtime_env.constants import (
@classmethod
def _load_schemas_from_env_var(cls):
    schema_paths = os.environ.get(RAY_RUNTIME_ENV_PLUGIN_SCHEMAS_ENV_VAR)
    if schema_paths:
        schema_json_files = list()
        for path in schema_paths.split(','):
            if path.endswith(RAY_RUNTIME_ENV_PLUGIN_SCHEMA_SUFFIX):
                schema_json_files.append(path)
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for f in files:
                        if f.endswith(RAY_RUNTIME_ENV_PLUGIN_SCHEMA_SUFFIX):
                            schema_json_files.append(os.path.join(root, f))
        logger.info(f'Loading the runtime env schemas from env var: {schema_json_files}.')
        cls._load_schemas(schema_json_files)