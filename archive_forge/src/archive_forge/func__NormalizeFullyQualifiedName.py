import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def _NormalizeFullyQualifiedName(name):
    """Remove leading period from fully-qualified type name.

  Due to b/13860351 in descriptor_database.py, types in the root namespace are
  generated with a leading period. This function removes that prefix.

  Args:
    name (str): The fully-qualified symbol name.

  Returns:
    str: The normalized fully-qualified symbol name.
  """
    return name.lstrip('.')