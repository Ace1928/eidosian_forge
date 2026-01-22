from datetime import datetime
import json
import logging
import numpy as np
import os
from urllib.parse import urlparse
import time
from ray.air._internal.json import SafeFallbackEncoder
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.offline.io_context import IOContext
from ray.rllib.offline.output_writer import OutputWriter
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.compression import pack, compression_supported
from ray.rllib.utils.typing import FileType, SampleBatchType
from typing import Any, Dict, List
def _to_json_dict(batch: SampleBatchType, compress_columns: List[str]) -> Dict:
    out = {}
    if isinstance(batch, MultiAgentBatch):
        out['type'] = 'MultiAgentBatch'
        out['count'] = batch.count
        policy_batches = {}
        for policy_id, sub_batch in batch.policy_batches.items():
            policy_batches[policy_id] = {}
            for k, v in sub_batch.items():
                policy_batches[policy_id][k] = _to_jsonable(v, compress=k in compress_columns)
        out['policy_batches'] = policy_batches
    else:
        out['type'] = 'SampleBatch'
        for k, v in batch.items():
            out[k] = _to_jsonable(v, compress=k in compress_columns)
    return out