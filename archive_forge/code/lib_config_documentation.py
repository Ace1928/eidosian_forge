from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import importlib
import logging
import os
import sys
import threading
from googlecloudsdk.core.util import encoding
Dynamic attribute access.

    Args:
      suffix: The attribute name.

    Returns:
      A configuration values.

    Raises:
      AttributeError: If the suffix is not a registered suffix.

    The first time an attribute is referenced, this method is invoked. The value
    returned is taken either from the config module or from the registered
    default.
    