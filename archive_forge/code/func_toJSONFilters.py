import codecs
import hashlib
import io
import json
import os
import sys
import atexit
import shutil
import tempfile
def toJSONFilters(actions):
    """Generate a JSON-to-JSON filter from stdin to stdout

    The filter:

    * reads a JSON-formatted pandoc document from stdin
    * transforms it by walking the tree and performing the actions
    * returns a new JSON-formatted pandoc document to stdout

    The argument `actions` is a list of functions of the form
    `action(key, value, format, meta)`, as described in more
    detail under `walk`.

    This function calls `applyJSONFilters`, with the `format`
    argument provided by the first command-line argument,
    if present.  (Pandoc sets this by default when calling
    filters.)
    """
    try:
        input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    except AttributeError:
        input_stream = codecs.getreader('utf-8')(sys.stdin)
    source = input_stream.read()
    if len(sys.argv) > 1:
        format = sys.argv[1]
    else:
        format = ''
    sys.stdout.write(applyJSONFilters(actions, source, format))