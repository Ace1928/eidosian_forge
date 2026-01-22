from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _register_arrow_json_readoptions_serializer(serialization_context):
    if os.environ.get(RAY_DISABLE_CUSTOM_ARROW_JSON_OPTIONS_SERIALIZATION, '0') == '1':
        return
    import pyarrow.json as pajson
    serialization_context._register_cloudpickle_serializer(pajson.ReadOptions, custom_serializer=lambda opts: (opts.use_threads, opts.block_size), custom_deserializer=lambda args: pajson.ReadOptions(*args))