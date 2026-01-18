import json
import logging
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
import requests
from mlflow.deployments import PredictionsResponse
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import scoring_server
from mlflow.utils.proto_json_utils import dump_input_data
def wait_server_ready(self, timeout=30, scoring_server_proc=None):
    return_code = self.process.poll()
    if return_code is not None:
        raise RuntimeError(f'Server process already exit with returncode {return_code}')