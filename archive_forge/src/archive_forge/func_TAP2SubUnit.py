import os
import re
import subprocess
import sys
import unittest
from io import BytesIO
from io import UnsupportedOperation as _UnsupportedOperation
import iso8601
from testtools import ExtendedToOriginalDecorator, content, content_type
from testtools.compat import _b, _u
from testtools.content import TracebackContent
from testtools import CopyStreamResult, testresult
from subunit import chunked, details
from subunit.v2 import ByteStreamToStreamResult, StreamResultToBytes
def TAP2SubUnit(tap, output_stream):
    """Filter a TAP pipe into a subunit pipe.

    This should be invoked once per TAP script, as TAP scripts get
    mapped to a single runnable case with multiple components.

    :param tap: A tap pipe/stream/file object - should emit unicode strings.
    :param subunit: A pipe/stream/file object to write subunit results to.
    :return: The exit code to exit with.
    """
    output = StreamResultToBytes(output_stream)
    UTF8_TEXT = 'text/plain; charset=UTF8'
    BEFORE_PLAN = 0
    AFTER_PLAN = 1
    SKIP_STREAM = 2
    state = BEFORE_PLAN
    plan_start = 1
    plan_stop = 0
    test_name = None
    log = []
    result = None

    def missing_test(plan_start):
        output.status(test_id='test %d' % plan_start, test_status='fail', runnable=False, mime_type=UTF8_TEXT, eof=True, file_name='tap meta', file_bytes=b'test missing from TAP output')

    def _emit_test():
        """write out a test"""
        if test_name is None:
            return
        if log:
            log_bytes = b'\n'.join((log_line.encode('utf8') for log_line in log))
            mime_type = UTF8_TEXT
            file_name = 'tap comment'
            eof = True
        else:
            log_bytes = None
            mime_type = None
            file_name = None
            eof = True
        del log[:]
        output.status(test_id=test_name, test_status=result, file_bytes=log_bytes, mime_type=mime_type, eof=eof, file_name=file_name, runnable=False)
    for line in tap:
        if state == BEFORE_PLAN:
            match = re.match('(\\d+)\\.\\.(\\d+)\\s*(?:\\#\\s+(.*))?\\n', line)
            if match:
                state = AFTER_PLAN
                _, plan_stop, comment = match.groups()
                plan_stop = int(plan_stop)
                if plan_start > plan_stop and plan_stop == 0:
                    state = SKIP_STREAM
                    output.status(test_id='file skip', test_status='skip', file_bytes=comment.encode('utf8'), eof=True, file_name='tap comment')
                continue
        match = re.match('(ok|not ok)(?:\\s+(\\d+)?)?(?:\\s+([^#]*[^#\\s]+)\\s*)?(?:\\s+#\\s+(TODO|SKIP|skip|todo)(?:\\s+(.*))?)?\\n', line)
        if match:
            _emit_test()
            status, number, description, directive, directive_comment = match.groups()
            if status == 'ok':
                result = 'success'
            else:
                result = 'fail'
            if description is None:
                description = ''
            else:
                description = ' ' + description
            if directive is not None:
                if directive.upper() == 'TODO':
                    result = 'xfail'
                elif directive.upper() == 'SKIP':
                    result = 'skip'
                if directive_comment is not None:
                    log.append(directive_comment)
            if number is not None:
                number = int(number)
                while plan_start < number:
                    missing_test(plan_start)
                    plan_start += 1
            test_name = 'test %d%s' % (plan_start, description)
            plan_start += 1
            continue
        match = re.match('Bail out\\!(?:\\s*(.*))?\\n', line)
        if match:
            reason, = match.groups()
            if reason is None:
                extra = ''
            else:
                extra = ' %s' % reason
            _emit_test()
            test_name = 'Bail out!%s' % extra
            result = 'fail'
            state = SKIP_STREAM
            continue
        match = re.match('\\#.*\\n', line)
        if match:
            log.append(line[:-1])
            continue
        output.status(file_bytes=line.encode('utf8'), file_name='stdout', mime_type=UTF8_TEXT)
    _emit_test()
    while plan_start <= plan_stop:
        missing_test(plan_start)
        plan_start += 1
    return 0