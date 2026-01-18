from typing import List, Optional, Tuple, Union
import pyarrow as pa
import pyhdk
from packaging import version
from pyhdk.hdk import HDK, ExecutionResult, QueryNode, RelAlgExecutor
from modin.config import CpuCount, HdkFragmentSize, HdkLaunchParameters
from modin.utils import _inherit_docstrings
from .base_worker import BaseDbWorker, DbTable

        Return the initialized HDK instance.

        Returns
        -------
        HDK
        