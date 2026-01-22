from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import errno
import hashlib
import json
import os
import re
import sys
import six
from boto import config
from gslib.exception import CommandException
from gslib.utils.boto_util import GetGsutilStateDir
from gslib.utils.boto_util import ResumableThreshold
from gslib.utils.constants import UTF8
from gslib.utils.hashing_helper import GetMd5
from gslib.utils.system_util import CreateDirIfNeeded
def HashRewriteParameters(src_obj_metadata, dst_obj_metadata, projection, src_generation=None, gen_match=None, meta_gen_match=None, canned_acl=None, max_bytes_per_call=None, src_dec_key_sha256=None, dst_enc_key_sha256=None, fields=None):
    """Creates an MD5 hex digest of the parameters for a rewrite call.

  Resuming rewrites requires that the input parameters are identical. Thus,
  the rewrite tracker file needs to represent the input parameters. For
  easy comparison, hash the input values. If a user does a performs a
  same-source/same-destination rewrite via a different command (for example,
  with a changed ACL), the hashes will not match and we will restart the
  rewrite from the beginning.

  Args:
    src_obj_metadata: apitools Object describing source object. Must include
      bucket, name, and etag.
    dst_obj_metadata: apitools Object describing destination object. Must
      include bucket and object name
    projection: Projection used for the API call.
    src_generation: Optional source generation.
    gen_match: Optional generation precondition.
    meta_gen_match: Optional metageneration precondition.
    canned_acl: Optional canned ACL string.
    max_bytes_per_call: Optional maximum bytes rewritten per call.
    src_dec_key_sha256: Optional SHA256 hash string of decryption key for
        source object.
    dst_enc_key_sha256: Optional SHA256 hash string of encryption key for
        destination object.
    fields: Optional fields to include in response to call.

  Returns:
    MD5 hex digest Hash of the input parameters, or None if required parameters
    are missing.
  """
    if not src_obj_metadata or not src_obj_metadata.bucket or (not src_obj_metadata.name) or (not src_obj_metadata.etag) or (not dst_obj_metadata) or (not dst_obj_metadata.bucket) or (not dst_obj_metadata.name) or (not projection):
        return
    md5_hash = GetMd5()
    for input_param in (src_obj_metadata, dst_obj_metadata, projection, src_generation, gen_match, meta_gen_match, canned_acl, fields, max_bytes_per_call, src_dec_key_sha256, dst_enc_key_sha256):
        if input_param is not None:
            md5_hash.update(six.text_type(input_param).encode('UTF8'))
    return md5_hash.hexdigest()