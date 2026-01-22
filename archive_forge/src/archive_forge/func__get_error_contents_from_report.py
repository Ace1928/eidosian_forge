from _pydev_runfiles import pydev_runfiles_xml_rpc
import pickle
import zlib
import base64
import os
from pydevd_file_utils import canonical_normalized_path
import pytest
import sys
import time
from pathlib import Path
def _get_error_contents_from_report(report):
    if report.longrepr is not None:
        try:
            tw = TerminalWriter(stringio=True)
            stringio = tw.stringio
        except TypeError:
            import io
            stringio = io.StringIO()
            tw = TerminalWriter(file=stringio)
        tw.hasmarkup = False
        report.toterminal(tw)
        exc = stringio.getvalue()
        s = exc.strip()
        if s:
            return s
    return ''