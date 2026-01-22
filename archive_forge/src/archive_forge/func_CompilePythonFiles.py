from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import compileall
import errno
import logging
import os
import posixpath
import re
import shutil
import sys
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import snapshots
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
import six
def CompilePythonFiles(self, force=False, workers=None):
    """Attempts to compile all the python files into .pyc files.

    Args:
      force: boolean, passed to force option of compileall.compiledir,
      workers: int, can be used to explicitly set number of worker processes;
        otherwise we determine it automatically. Only set for testing.

    This does not raise exceptions if compiling a given file fails.
    """
    if six.PY2:
        regex_exclusion = re.compile('(httplib2/python3|typing/python3|platform/bq/third_party/yaml/lib3|third_party/google/api_core|third_party/google/auth|third_party/google/oauth2|third_party/overrides|third_party/proto|dulwich|gapic|pubsublite|pubsub/lite_subscriptions.py|logging_v2|platform/bundledpythonunix|pubsub_v1/services)')
    elif sys.version_info[1] == 4:
        regex_exclusion = re.compile('.*')
    elif sys.version_info[1] >= 7:
        regex_exclusion = re.compile('(kubernetes/utils/create_from_yaml.py|platform/google_appengine|gslib/vendored/boto/boto/iam/connection.py|gslib/vendored/boto/tests/|third_party/.*/python2/|third_party/yaml/[a-z]*.py|third_party/yaml/lib2/|third_party/appengine/|third_party/fancy_urllib/|platform/bq/third_party/gflags|platform/ext-runtime/nodejs/test/|platform/gsutil/third_party/apitools/ez_setup|platform/gsutil/third_party/crcmod_osx/crcmod/test)')
    else:
        regex_exclusion = None
    with file_utils.ChDir(self.sdk_root):
        to_compile = [os.path.join('bin', 'bootstrapping'), os.path.join('data', 'cli'), 'lib', 'platform']
        num_workers = min(os.cpu_count(), 8) if workers is None else workers
        for d in to_compile:
            compileall.compile_dir(d, rx=regex_exclusion, quiet=2, force=force, workers=num_workers)