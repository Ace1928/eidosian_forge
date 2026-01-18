import importlib
import json
import os
from pathlib import Path
import re
import sys
import typer
from typing import Optional
import uuid
import yaml
import ray
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.resources import resources_to_json, json_to_resources
from ray.tune.tune import run_experiments
from ray.tune.schedulers import create_scheduler
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.common import CLIArguments as cli
from ray.rllib.common import FrameworkEnum, SupportedFileType
from ray.rllib.common import download_example_file, get_file_type
Run the CLI.