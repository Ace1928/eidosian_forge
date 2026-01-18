from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
from googlecloudsdk.third_party.appengine.tools import context_util
import six
[DEPRECATED] Saves repository information in a file.

  DEPRECATED, use `gcloud beta debug source gen-repo-info-file` instead.  The
  generated file is an opaque blob representing which source revision the
  application was built at, and which Google-hosted repository this revision
  will be pushed to.
  