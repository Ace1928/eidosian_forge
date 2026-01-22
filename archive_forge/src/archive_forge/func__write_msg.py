import json
import os
import sys
def _write_msg(**message):
    """Write a message to standard output.

    Args:
        **message: ({str: object, ...}) A JSON message encoded in keyword
            arguments.
    """
    json.dump(message, sys.stdout, default=_make_serializable)
    sys.stdout.write('\n')
    sys.stdout.flush()