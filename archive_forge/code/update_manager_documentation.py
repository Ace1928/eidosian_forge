from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import hashlib
import itertools
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
import certifi
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import release_notes
from googlecloudsdk.core.updater import snapshots
from googlecloudsdk.core.updater import update_check
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
from six.moves import map  # pylint: disable=redefined-builtin
Runs the gcloud command to post process the update.

    This runs gcloud as a subprocess so that the new version of gcloud (the one
    we just updated to) is run instead of the old code (which is running here).
    We do this so the new code can say how to correctly post process itself.

    Args:
      snapshot: ComponentSnapshot, The component snapshot for the version
        we are updating do. The location of gcloud and the command to run can
        change from version to version, which is why we try to pull this
        information from the latest snapshot.  For a restore operation, we don't
        have that information so we fall back to a best effort default.
    