import os
from absl import flags
Returns the executable path of a py_binary.

  This returns the executable path of a py_binary that is in another Bazel
  target's data dependencies.

  On Linux/macOS, the path and __file__ has the same root directory.
  On Windows, bazel builds an .exe file and we need to use the MANIFEST file
  the location the actual binary.

  Args:
    py_binary_name: string, the name of a py_binary that is in another Bazel
        target's data dependencies.

  Raises:
    RuntimeError: Raised when it cannot locate the executable path.
  