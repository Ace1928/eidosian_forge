import copy
import yaml
import json
import os
import socket
import sys
import time
import threading
import logging
import uuid
import warnings
import requests
from packaging.version import Version
from typing import Optional, Dict, Tuple, Type
import ray
import ray._private.services
from ray.autoscaler._private.spark.node_provider import HEAD_NODE_ID
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray._private.storage import _load_class
from .utils import (
from .start_hook_base import RayOnSparkStartHook
from .databricks_hook import DefaultDatabricksRayOnSparkStartHook
def _check_system_environment():
    if os.name != 'posix':
        raise RuntimeError('Ray on spark only supports running on POSIX system.')
    spark_dependency_error = 'ray.util.spark module requires pyspark >= 3.3'
    try:
        import pyspark
        if Version(pyspark.__version__).release < (3, 3, 0):
            raise RuntimeError(spark_dependency_error)
    except ImportError:
        raise RuntimeError(spark_dependency_error)