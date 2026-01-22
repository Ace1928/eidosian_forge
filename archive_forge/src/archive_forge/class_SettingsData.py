import collections.abc
import configparser
import enum
import getpass
import json
import logging
import multiprocessing
import os
import platform
import re
import shutil
import socket
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from distutils.util import strtobool
from functools import reduce
from typing import (
from urllib.parse import quote, unquote, urlencode, urlparse, urlsplit
from google.protobuf.wrappers_pb2 import BoolValue, DoubleValue, Int32Value, StringValue
import wandb
import wandb.env
from wandb import util
from wandb.apis.internal import Api
from wandb.errors import UsageError
from wandb.proto import wandb_settings_pb2
from wandb.sdk.internal.system.env_probe_helpers import is_aws_lambda
from wandb.sdk.lib import filesystem
from wandb.sdk.lib._settings_toposort_generated import SETTINGS_TOPOLOGICALLY_SORTED
from wandb.sdk.wandb_setup import _EarlyLogger
from .lib import apikey
from .lib.gitlib import GitRepo
from .lib.ipython import _get_python_type
from .lib.runid import generate_id
@dataclass()
class SettingsData:
    """Settings for the W&B SDK."""
    _args: Sequence[str]
    _aws_lambda: bool
    _async_upload_concurrency_limit: int
    _cli_only_mode: bool
    _code_path_local: str
    _colab: bool
    _cuda: str
    _disable_meta: bool
    _disable_service: bool
    _disable_setproctitle: bool
    _disable_stats: bool
    _disable_viewer: bool
    _disable_machine_info: bool
    _except_exit: bool
    _executable: str
    _extra_http_headers: Mapping[str, str]
    _file_stream_retry_max: int
    _file_stream_retry_wait_min_seconds: float
    _file_stream_retry_wait_max_seconds: float
    _file_stream_timeout_seconds: float
    _file_transfer_retry_max: int
    _file_transfer_retry_wait_min_seconds: float
    _file_transfer_retry_wait_max_seconds: float
    _file_transfer_timeout_seconds: float
    _flow_control_custom: bool
    _flow_control_disabled: bool
    _graphql_retry_max: int
    _graphql_retry_wait_min_seconds: float
    _graphql_retry_wait_max_seconds: float
    _graphql_timeout_seconds: float
    _internal_check_process: float
    _internal_queue_timeout: float
    _ipython: bool
    _jupyter: bool
    _jupyter_name: str
    _jupyter_path: str
    _jupyter_root: str
    _kaggle: bool
    _live_policy_rate_limit: int
    _live_policy_wait_time: int
    _log_level: int
    _network_buffer: int
    _noop: bool
    _notebook: bool
    _offline: bool
    _sync: bool
    _os: str
    _platform: str
    _proxies: Mapping[str, str]
    _python: str
    _runqueue_item_id: str
    _require_core: bool
    _save_requirements: bool
    _service_transport: str
    _service_wait: float
    _shared: bool
    _start_datetime: str
    _start_time: float
    _stats_pid: int
    _stats_sample_rate_seconds: float
    _stats_samples_to_average: int
    _stats_join_assets: bool
    _stats_neuron_monitor_config_path: str
    _stats_open_metrics_endpoints: Mapping[str, str]
    _stats_open_metrics_filters: Union[Sequence[str], Mapping[str, Mapping[str, str]]]
    _stats_disk_paths: Sequence[str]
    _stats_buffer_size: int
    _tmp_code_dir: str
    _tracelog: str
    _unsaved_keys: Sequence[str]
    _windows: bool
    allow_val_change: bool
    anonymous: str
    api_key: str
    azure_account_url_to_access_key: Dict[str, str]
    base_url: str
    code_dir: str
    colab_url: str
    config_paths: Sequence[str]
    console: str
    deployment: str
    disable_code: bool
    disable_git: bool
    disable_hints: bool
    disable_job_creation: bool
    disabled: bool
    docker: str
    email: str
    entity: str
    files_dir: str
    force: bool
    git_commit: str
    git_remote: str
    git_remote_url: str
    git_root: str
    heartbeat_seconds: int
    host: str
    ignore_globs: Tuple[str]
    init_timeout: float
    is_local: bool
    job_name: str
    job_source: str
    label_disable: bool
    launch: bool
    launch_config_path: str
    log_dir: str
    log_internal: str
    log_symlink_internal: str
    log_symlink_user: str
    log_user: str
    login_timeout: float
    mode: str
    notebook_name: str
    problem: str
    program: str
    program_abspath: str
    program_relpath: str
    project: str
    project_url: str
    quiet: bool
    reinit: bool
    relogin: bool
    resume: Union[str, bool]
    resume_fname: str
    resumed: bool
    root_dir: str
    run_group: str
    run_id: str
    run_job_type: str
    run_mode: str
    run_name: str
    run_notes: str
    run_tags: Tuple[str]
    run_url: str
    sagemaker_disable: bool
    save_code: bool
    settings_system: str
    settings_workspace: str
    show_colors: bool
    show_emoji: bool
    show_errors: bool
    show_info: bool
    show_warnings: bool
    silent: bool
    start_method: str
    strict: bool
    summary_errors: int
    summary_timeout: int
    summary_warnings: int
    sweep_id: str
    sweep_param_path: str
    sweep_url: str
    symlink: bool
    sync_dir: str
    sync_file: str
    sync_symlink_latest: str
    system_sample: int
    system_sample_seconds: int
    table_raise_on_max_row_limit_exceeded: bool
    timespec: str
    tmp_dir: str
    username: str
    wandb_dir: str