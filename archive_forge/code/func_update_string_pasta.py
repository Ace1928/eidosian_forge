import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
def update_string_pasta(self, text, in_filename):
    """Updates a file using pasta."""
    try:
        t = pasta.parse(text)
    except (SyntaxError, ValueError, TypeError):
        log = ['ERROR: Failed to parse.\n' + traceback.format_exc()]
        return (0, '', log, [])
    t, preprocess_logs, preprocess_errors = self._api_change_spec.preprocess(t)
    visitor = _PastaEditVisitor(self._api_change_spec)
    visitor.visit(t)
    self._api_change_spec.clear_preprocessing()
    logs = [self.format_log(log, None) for log in preprocess_logs + visitor.log]
    errors = [self.format_log(error, in_filename) for error in preprocess_errors + visitor.warnings_and_errors]
    return (1, pasta.dump(t), logs, errors)