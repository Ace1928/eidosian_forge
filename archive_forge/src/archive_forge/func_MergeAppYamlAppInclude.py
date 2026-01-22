from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
@classmethod
def MergeAppYamlAppInclude(cls, appyaml, appinclude):
    """Merges an `app.yaml` file with referenced builtins/includes.

    Args:
      appyaml: The `app.yaml` file that you want to update with `appinclude`.
      appinclude: The includes that you want to merge into `appyaml`.

    Returns:
      An updated `app.yaml` file that includes the directives you specified in
      `appinclude`.
    """
    if not appinclude:
        return appyaml
    if appinclude.handlers:
        tail = appyaml.handlers or []
        appyaml.handlers = []
        for h in appinclude.handlers:
            if not h.position or h.position == 'head':
                appyaml.handlers.append(h)
            else:
                tail.append(h)
            h.position = None
        appyaml.handlers.extend(tail)
    appyaml = cls._CommonMergeOps(appyaml, appinclude)
    appyaml.NormalizeVmSettings()
    return appyaml