import logging
from typing import Optional, Tuple
import concurrent.futures
import ray.dashboard.modules.log.log_utils as log_utils
import ray.dashboard.modules.log.log_consts as log_consts
import ray.dashboard.utils as dashboard_utils
import ray.dashboard.optional_utils as dashboard_optional_utils
from ray._private.ray_constants import env_integer
import asyncio
import grpc
import io
import os
from pathlib import Path
from ray.core.generated import reporter_pb2
from ray.core.generated import reporter_pb2_grpc
from ray._private.ray_constants import (

        Streams the log in real time starting from `request.lines` number of lines from
        the end of the file if `request.keep_alive == True`. Else, it terminates the
        stream once there are no more bytes to read from the log file.

        Part of `LogService` gRPC.

        NOTE: These RPCs are used by state_head.py, not log_head.py
        