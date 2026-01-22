from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import errno
import os
import re
import signal
import subprocess
import sys
import threading
import time
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import parallel
from googlecloudsdk.core.util import platforms
import six
from six.moves import map
def GetToolEnv(env=None):
    """Generate the environment that should be used for the subprocess.

  Args:
    env: {str, str}, An existing environment to augment.  If None, the current
      environment will be cloned and used as the base for the subprocess.

  Returns:
    The modified env.
  """
    if env is None:
        env = dict(os.environ)
    env = encoding.EncodeEnv(env)
    encoding.SetEncodedValue(env, 'CLOUDSDK_WRAPPER', '1')
    for s in properties.VALUES:
        for p in s:
            if p.is_feature_flag:
                continue
            encoding.SetEncodedValue(env, p.EnvironmentName(), p.Get(required=False, validate=False))
    encoding.SetEncodedValue(env, config.CLOUDSDK_ACTIVE_CONFIG_NAME, named_configs.ConfigurationStore.ActiveConfig().name)
    return env