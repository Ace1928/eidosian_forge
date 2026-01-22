from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.transfer import jobs_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.transfer import name_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import scaled_integer
def _print_progress(operation, retryer_state):
    """Gets operation from API and prints its progress updating in-place."""
    metadata = encoding.MessageToDict(operation.metadata)
    if 'counters' in metadata:
        skipped_bytes = int(metadata['counters'].get('bytesFromSourceSkippedBySync', 0))
        skipped_string = scaled_integer.FormatBinaryNumber(skipped_bytes, decimal_places=1)
        copied_bytes = int(metadata['counters'].get('bytesCopiedToSink', 0))
        total_bytes = int(metadata['counters'].get('bytesFoundFromSource', 0))
        if total_bytes:
            progress_percent = int(round(copied_bytes / total_bytes, 2) * 100)
        else:
            progress_percent = 0
        progress_string = '{}% ({} of {})'.format(progress_percent, scaled_integer.FormatBinaryNumber(copied_bytes, decimal_places=1), scaled_integer.FormatBinaryNumber(total_bytes, decimal_places=1))
    else:
        progress_string = 'Progress: {}'.format(_UNKNOWN_VALUE)
        skipped_string = _UNKNOWN_VALUE
    if 'errorBreakdowns' in metadata:
        error_count = sum([int(error['errorCount']) for error in metadata['errorBreakdowns']])
    else:
        error_count = 0
    spin_marks = console_attr.ProgressTrackerSymbolsAscii().spin_marks
    if retryer_state.retrial == _LAST_RETRIAL:
        spin_mark = ''
    else:
        spin_mark = spin_marks[retryer_state.retrial % len(spin_marks)]
    log.status.write('{} | {} | Skipped: {} | Errors: {} {}\r'.format(metadata['status'], progress_string, skipped_string, error_count, spin_mark))