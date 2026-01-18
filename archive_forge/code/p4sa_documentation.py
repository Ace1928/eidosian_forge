from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import iam as kms_iam
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.api_lib.storage import storage_api
Adds the necessary P4SA role bindings on the given key and bucket.

  Args:
    p4sa_email: Email address of the P4SA for which to add role bindings. This
                can come from a call to GetOrCreate().
    kms_key_ref: optional, resources.Resource reference to the KMS key on which
                 to add a role binding.
    bucket_ref: optional, storage_util.BucketReference to the GCS bucket on
                which to add a role binding.
  