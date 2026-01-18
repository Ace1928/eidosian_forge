from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import files
Digest the file_to_digest based on digest_algorithm.

  Args:
    digest_algorithm: The algorithm used to digest the file, can be one of
      'sha256', 'sha384', or 'sha512'.
    file_to_digest: A valid file handle.

  Returns:
    The digest of the provided file.

  Raises:
    InvalidArgumentException: The provided digest_algorithm is invalid.
  