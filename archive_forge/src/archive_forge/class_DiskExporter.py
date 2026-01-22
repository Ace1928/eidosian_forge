import io
import json
import optparse
import os.path
import sys
from errno import EEXIST
from textwrap import dedent
from testtools import StreamToDict
from subunit.filters import run_tests_from_stream
class DiskExporter:
    """Exports tests to disk."""

    def __init__(self, directory):
        self._directory = os.path.realpath(directory)

    def export(self, test_dict):
        id = test_dict['id']
        tags = sorted(test_dict['tags'])
        details = test_dict['details']
        status = test_dict['status']
        start, stop = test_dict['timestamps']
        test_summary = {}
        test_summary['id'] = id
        test_summary['tags'] = tags
        test_summary['status'] = status
        test_summary['details'] = sorted(details.keys())
        test_summary['start'] = _json_time(start)
        test_summary['stop'] = _json_time(stop)
        root = _allocate_path(self._directory, id)
        with _open_path(root, 'test.json') as f:
            maybe_str = json.dumps(test_summary, sort_keys=True, ensure_ascii=False)
            if not isinstance(maybe_str, bytes):
                maybe_str = maybe_str.encode('utf-8')
            f.write(maybe_str)
        for name, detail in details.items():
            with _open_path(root, name) as f:
                for chunk in detail.iter_bytes():
                    f.write(chunk)