import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
def _parse_stack_frames_blob_key(blob_key):
    """Parse the BLOB key for source file list.

    Args:
      blob_key: The BLOB key to parse. By contract, it should have the format:
       `${STACK_FRAMES_BLOB_TAG_PREFIX}_` +
       `${stack_frame_id_0}_..._${stack_frame_id_N}.${run_id}`

    Returns:
      - run ID
      - The stack frame IDs as a tuple of strings.
    """
    key_body, run = blob_key.split('.', 1)
    key_body = key_body[len(STACK_FRAMES_BLOB_TAG_PREFIX) + 1:]
    stack_frame_ids = key_body.split('_')
    return (run, stack_frame_ids)