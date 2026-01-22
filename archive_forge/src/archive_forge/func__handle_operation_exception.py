from __future__ import absolute_import, division, print_function
import sys
import json
import re
import traceback as trace
def _handle_operation_exception(module, exception, traceback, verbosity):
    op = exception.op
    error_message = format_failed_fusion_operation_exception(exception)
    if verbosity > 1:
        module.fail_json(msg=error_message, op_details=op.to_dict(), traceback=str(traceback))
    elif verbosity > 0:
        module.fail_json(msg=error_message, op_details=op.to_dict())
    else:
        module.fail_json(msg=error_message)