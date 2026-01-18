from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import os
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
import six
Parse flags from app engine dev_appserver.py.

  Only the subset of args are parsed here. These args are listed in
  _UPSTREAM_DEV_APPSERVER_FLAGS.

  Args:
    args: A list of arguments (typically sys.argv).

  Returns:
    options: An argparse.Namespace containing the command line arguments.
  