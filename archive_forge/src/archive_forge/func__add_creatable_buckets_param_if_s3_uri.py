import asyncio
import binascii
from collections import defaultdict
import contextlib
import errno
import functools
import importlib
import inspect
import json
import logging
import multiprocessing
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlencode, unquote, urlparse, parse_qsl, urlunparse
import warnings
from inspect import signature
from pathlib import Path
from subprocess import list2cmdline
from typing import (
import psutil
from google.protobuf import json_format
import ray
import ray._private.ray_constants as ray_constants
from ray.core.generated.runtime_env_common_pb2 import (
def _add_creatable_buckets_param_if_s3_uri(uri: str) -> str:
    """If the provided URI is an S3 URL, add allow_bucket_creation=true as a query
    parameter. For pyarrow >= 9.0.0, this is required in order to allow
    ``S3FileSystem.create_dir()`` to create S3 buckets.

    If the provided URI is not an S3 URL or if pyarrow < 9.0.0 is installed, we return
    the URI unchanged.

    Args:
        uri: The URI that we'll add the query parameter to, if it's an S3 URL.

    Returns:
        A URI with the added allow_bucket_creation=true query parameter, if the provided
        URI is an S3 URL; uri will be returned unchanged otherwise.
    """
    from pkg_resources._vendor.packaging.version import parse as parse_version
    pyarrow_version = _get_pyarrow_version()
    if pyarrow_version is not None:
        pyarrow_version = parse_version(pyarrow_version)
    if pyarrow_version is not None and pyarrow_version < parse_version('9.0.0'):
        return uri
    parsed_uri = urlparse(uri)
    if parsed_uri.scheme == 's3':
        uri = _add_url_query_params(uri, {'allow_bucket_creation': True})
    return uri