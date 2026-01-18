import io
import math
import os
import typing
import weakref
def xref_copy(doc: fitz.Document, source: int, target: int, *, keep: list=None) -> None:
    """Copy a PDF dictionary object to another one given their xref numbers.

    Args:
        doc: PDF document object
        source: source xref number
        target: target xref number, the xref must already exist
        keep: an optional list of 1st level keys in target that should not be
              removed before copying.
    Notes:
        This works similar to the copy() method of dictionaries in Python. The
        source may be a stream object.
    """
    if doc.xref_is_stream(source):
        stream = doc.xref_stream_raw(source)
        doc.update_stream(target, stream, compress=False, new=True)
    if keep is None:
        keep = []
    for key in doc.xref_get_keys(target):
        if key in keep:
            continue
        doc.xref_set_key(target, key, 'null')
    for key in doc.xref_get_keys(source):
        item = doc.xref_get_key(source, key)
        doc.xref_set_key(target, key, item[1])