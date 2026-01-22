import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
def _parse_execution_data_blob_key(blob_key):
    """Parse the BLOB key for Execution data objects.

    This differs from `_parse_execution_digest_blob_key()` in that it is
    for the deatiled data objects for execution, instead of the digests.

    Args:
      blob_key: The BLOB key to parse. By contract, it should have the format:
       `${EXECUTION_DATA_BLOB_TAG_PREFIX}_${begin}_${end}.${run_id}`

    Returns:
      - run ID
      - begin index
      - end index
    """
    key_body, run = blob_key.split('.', 1)
    key_body = key_body[len(EXECUTION_DATA_BLOB_TAG_PREFIX):]
    begin = int(key_body.split('_')[1])
    end = int(key_body.split('_')[2])
    return (run, begin, end)