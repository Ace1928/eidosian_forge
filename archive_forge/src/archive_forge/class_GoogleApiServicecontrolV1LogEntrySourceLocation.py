from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServicecontrolV1LogEntrySourceLocation(_messages.Message):
    """Additional information about the source code location that produced the
  log entry.

  Fields:
    file: Optional. Source file name. Depending on the runtime environment,
      this might be a simple name or a fully-qualified name.
    function: Optional. Human-readable name of the function or method being
      invoked, with optional context such as the class or package name. This
      information may be used in contexts such as the logs viewer, where a
      file and line number are less meaningful. The format can vary by
      language. For example: `qual.if.ied.Class.method` (Java),
      `dir/package.func` (Go), `function` (Python).
    line: Optional. Line within the source file. 1-based; 0 indicates no line
      number available.
  """
    file = _messages.StringField(1)
    function = _messages.StringField(2)
    line = _messages.IntegerField(3)