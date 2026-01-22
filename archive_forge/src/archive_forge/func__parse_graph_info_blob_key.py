import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
def _parse_graph_info_blob_key(blob_key):
    """Parse the BLOB key for graph info.

    Args:
      blob_key: The BLOB key to parse. By contract, it should have the format:
       `${GRAPH_INFO_BLOB_TAG_PREFIX}_${graph_id}.${run_name}`,

    Returns:
      - run name
      - graph_id
    """
    key_body, run = blob_key.split('.')
    graph_id = key_body[len(GRAPH_INFO_BLOB_TAG_PREFIX) + 1:]
    return (run, graph_id)