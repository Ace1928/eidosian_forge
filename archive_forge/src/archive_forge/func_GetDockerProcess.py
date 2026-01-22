from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import json
import os
import subprocess
import sys
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import semver
import six
from six.moves import urllib
@EnsureDocker
def GetDockerProcess(docker_args, stdin_file, stdout_file, stderr_file):
    return subprocess.Popen(['docker'] + docker_args, stdin=stdin_file, stdout=stdout_file, stderr=stderr_file)