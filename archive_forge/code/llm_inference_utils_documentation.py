from __future__ import annotations
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.types.llm import (

    Get corresponding original Transformers task for the given LLM inference task.

    NB: This assumes there is only one original Transformers task for each LLM inference
      task, which might not be true in the future.
    