import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
def _parse_source_file_blob_key(blob_key):
    """Parse the BLOB key for accessing the content of a source file.

    Args:
      blob_key: The BLOB key to parse. By contract, it should have the format:
       `${SOURCE_FILE_BLOB_TAG_PREFIX}_${index}.${run_id}`

    Returns:
      - run ID, as a str.
      - File index, as an int.
    """
    key_body, run = blob_key.split('.', 1)
    index = int(key_body[len(SOURCE_FILE_BLOB_TAG_PREFIX) + 1:])
    return (run, index)