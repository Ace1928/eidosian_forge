from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import runtime_registry
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.command_lib.util import gcloudignore
from googlecloudsdk.core import exceptions as core_exceptions
Returns an iterator for accessing all source files to be uploaded.

  This method uses several implementations based on the provided runtime and
  env. The rules are as follows, in decreasing priority:
  1) For some runtimes/envs (i.e. those defined in _GCLOUDIGNORE_REGISTRY), we
     completely ignore skip_files and generate a runtime-specific .gcloudignore
     if one is not present, or use the existing .gcloudignore.
  2) For all other runtimes/envs, we:
    2a) If ignore_file is not none, use custom ignore_file to skip files. If the
        specified file does not exist, raise error. We also raise an error if
        the user has both ignore file and explicit skip_files defined.
    2b) If user does not specify ignore_file, check for an existing
        .gcloudignore and use that if one exists. We also raise an error if
        the user has both a .gcloudignore file and explicit skip_files defined.
    2c) If there is no .gcloudignore, we use the provided skip_files.

  Args:
    upload_dir: str, path to upload directory, the files to be uploaded.
    skip_files_regex: str, skip_files to use if necessary - see above rules for
      when this could happen. This can be either the user's explicit skip_files
      as defined in their app.yaml or the default skip_files we implicitly
      provide if they didn't define any.
    has_explicit_skip_files: bool, indicating whether skip_files_regex was
      explicitly defined by the user
    runtime: str, runtime as defined in app.yaml
    environment: env.Environment enum
    source_dir: str, path to original source directory, for writing generated
      files. May be the same as upload_dir.
    ignore_file: custom ignore_file name.
              Override .gcloudignore file to customize files to be skipped.

  Raises:
    SkipFilesError: if you are using a runtime that no longer supports
      skip_files (such as those defined in _GCLOUDIGNORE_REGISTRY), or if using
      a runtime that still supports skip_files, but both skip_files and
      a. gcloudignore file are present.
    FileNotFoundError: if the custom ignore-file does not exist.

  Returns:
    A list of path names of source files to be uploaded.
  