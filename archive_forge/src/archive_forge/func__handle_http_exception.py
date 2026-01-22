from __future__ import absolute_import, division, print_function
import sys
import json
import re
import traceback as trace
def _handle_http_exception(module, exception, traceback, verbosity):
    error_message = format_http_exception(exception, traceback)
    if verbosity > 1:
        module.fail_json(msg=error_message, traceback=trace.format_exception(exception))
    else:
        module.fail_json(msg=error_message)