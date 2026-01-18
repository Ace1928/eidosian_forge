from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
Performs a Firestore Admin v1 Import.

  Args:
    project: the project id to import, a string.
    database: the databae id to import, a string.
    input_uri_prefix: the input uri prefix of the exported files, a string.
    namespace_ids: a string list of namespace ids to import.
    collection_ids: a string list of collections to import.

  Returns:
    an Operation.
  