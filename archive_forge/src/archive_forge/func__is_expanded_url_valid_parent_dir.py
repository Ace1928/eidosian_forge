from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import path_util
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import posix_util
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.command_lib.storage.tasks.cp import copy_task_factory
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _is_expanded_url_valid_parent_dir(expanded_url):
    """Returns True if not FileUrl ending in  relative path symbols.

  A URL is invalid if it is a FileUrl and the parent directory of the file is a
  relative path symbol. Unix will not allow a file itself to be named with a
  relative path symbol, but one can be the parent. Notably, "../obj" can lead
  to unexpected behavior at the copy destination. We examine the pre-recursion
  expanded_url, which might point to "..", to see if the parent is valid.

  If the user does a recursive copy from an expanded URL, it may not end up
  the final parent of the copied object. For example, see: "dir/nested_dir/obj".

  If you ran "cp -r d* gs://bucket" from the parent of "dir", then the
  expanded_url would be "dir", but "nested_dir" would be the parent of "obj".
  This actually doesn't matter since recursion won't add relative path symbols
  to the path. However, we still return if expanded_url is valid because
  there are cases where we need to copy every parent directory up to
  expanded_url "dir" to prevent file name conflicts.

  Args:
    expanded_url (StorageUrl): NameExpansionResult.expanded_url value. Should
      contain wildcard-expanded URL before recursion. For example, if "d*"
      expands to the object "dir/obj", we would get the "dir" value.

  Returns:
    Boolean indicating if the expanded_url is valid as a parent
      directory.
  """
    if not isinstance(expanded_url, storage_url.FileUrl):
        return True
    _, _, last_string_following_delimiter = expanded_url.versionless_url_string.rstrip(expanded_url.delimiter).rpartition(expanded_url.delimiter)
    return last_string_following_delimiter not in _RELATIVE_PATH_SYMBOLS and last_string_following_delimiter not in [expanded_url.scheme.value + '://' + symbol for symbol in _RELATIVE_PATH_SYMBOLS]