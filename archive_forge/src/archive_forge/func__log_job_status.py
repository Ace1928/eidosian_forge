import json
import os
import sys
import pprint
import time
from subprocess import list2cmdline
from typing import Optional, Tuple, Union, Dict, Any
import click
import ray._private.ray_constants as ray_constants
from ray._private.storage import _load_class
from ray._private.utils import get_or_create_event_loop
from ray.autoscaler._private.cli_logger import add_click_logging_options, cf, cli_logger
from ray.dashboard.modules.dashboard_sdk import parse_runtime_env_args
from ray.job_submission import JobStatus, JobSubmissionClient
from ray.dashboard.modules.job.cli_utils import add_common_job_options
from ray.dashboard.modules.job.utils import redact_url_password
from ray.util.annotations import PublicAPI
from ray._private.utils import parse_resources_json, parse_metadata_json
def _log_job_status(client: JobSubmissionClient, job_id: str) -> JobStatus:
    info = client.get_job_info(job_id)
    if info.status == JobStatus.SUCCEEDED:
        _log_big_success_msg(f"Job '{job_id}' succeeded")
    elif info.status == JobStatus.STOPPED:
        cli_logger.warning(f"Job '{job_id}' was stopped")
    elif info.status == JobStatus.FAILED:
        _log_big_error_msg(f"Job '{job_id}' failed")
        if info.message is not None:
            cli_logger.print(f'Status message: {info.message}', no_format=True)
    else:
        cli_logger.print(f"Status for job '{job_id}': {info.status}")
        if info.message is not None:
            cli_logger.print(f'Status message: {info.message}', no_format=True)
    return info.status