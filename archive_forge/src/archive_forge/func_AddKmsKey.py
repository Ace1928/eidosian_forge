from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddKmsKey(parser, help_text='KMS key used to encrypt instance optionally.'):
    parser.add_argument('--kms-key', dest='kms_key', required=False, help=help_text)