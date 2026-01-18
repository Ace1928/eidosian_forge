import io
import json
import optparse
import os.path
import sys
from errno import EEXIST
from textwrap import dedent
from testtools import StreamToDict
from subunit.filters import run_tests_from_stream
def to_disk(argv=None, stdin=None, stdout=None):
    if stdout is None:
        stdout = sys.stdout
    if stdin is None:
        stdin = sys.stdin
    parser = optparse.OptionParser(description='Export a subunit stream to files on disk.', epilog=dedent('            Creates a directory per test id, a JSON file with test\n            metadata within that directory, and each attachment\n            is written to their name relative to that directory.\n\n            Global packages (no test id) are discarded.\n\n            Exits 0 if the export was completed, or non-zero otherwise.\n            '))
    parser.add_option('-d', '--directory', help='Root directory to export to.', default='.')
    options, args = parser.parse_args(argv)
    if len(args) > 1:
        raise Exception('Unexpected arguments.')
    if len(args):
        source = io.open(args[0], 'rb')
    else:
        source = stdin
    exporter = DiskExporter(options.directory)
    result = StreamToDict(exporter.export)
    run_tests_from_stream(source, result, protocol_version=2)
    return 0