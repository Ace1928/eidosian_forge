from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddLocalDiskKMSKey(parser):
    parser.add_argument('--local-disk-kms-key', help='\n      Google Cloud KMS key that will be used to secure local disks on nodes\n      in this node pool. The Edge Container service account for this project\n      must have `roles/cloudkms.cryptoKeyEncrypterDecrypter` on the key.\n\n      If not provided, a Google-managed key will be used instead.\n      ')