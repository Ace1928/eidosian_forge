from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import os
import tempfile
@contextlib.contextmanager
def NamedTempFile(contents, prefix='tmp', suffix='', delete=True):
    """Write a named temporary with given contents.

  Args:
    contents: (str) File contents.
    prefix: (str) File base name prefix.
    suffix: (str) Filename suffix.
    delete: (bool) Delete file on __exit__.

  Yields:
    The temporary file object.
  """
    common_args = dict(mode='w+t', prefix=prefix, suffix=suffix, delete=delete)
    if os.name == 'nt':
        with _WindowsNamedTempFile(**common_args) as f:
            f.write(contents)
            f.close()
            yield f
    else:
        with tempfile.NamedTemporaryFile(**common_args) as f:
            f.write(contents)
            f.flush()
            yield f