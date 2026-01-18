from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import subprocess
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
Require that a certain version of Java is installed.

  Args:
    for_text: str, the text explaining what Java is necessary for.
    min_version: int, the minimum major version to check for.

  Raises:
    JavaError: if a Java executable is not found or has the wrong version.

  Returns:
    str, Path to the Java executable.
  