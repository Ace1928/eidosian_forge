import codecs
import hashlib
import io
import json
import os
import sys
import atexit
import shutil
import tempfile
def applyJSONFilters(actions, source, format=''):
    """Walk through JSON structure and apply filters

    This:

    * reads a JSON-formatted pandoc document from a source string
    * transforms it by walking the tree and performing the actions
    * returns a new JSON-formatted pandoc document as a string

    The `actions` argument is a list of functions (see `walk`
    for a full description).

    The argument `source` is a string encoded JSON object.

    The argument `format` is a string describing the output format.

    Returns a the new JSON-formatted pandoc document.
    """
    doc = json.loads(source)
    if 'meta' in doc:
        meta = doc['meta']
    elif doc[0]:
        meta = doc[0]['unMeta']
    else:
        meta = {}
    altered = doc
    for action in actions:
        altered = walk(altered, action, format, meta)
    return json.dumps(altered)