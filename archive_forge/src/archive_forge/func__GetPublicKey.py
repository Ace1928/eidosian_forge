from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import cryptokeyversions
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.privateca import certificate_utils
from googlecloudsdk.api_lib.privateca import request_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.privateca import flags
from googlecloudsdk.command_lib.privateca import key_generation
from googlecloudsdk.command_lib.privateca import pem_utils
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
def _GetPublicKey(self, args):
    """Fetches the public key associated with a non-CSR certificate request, as UTF-8 encoded bytes."""
    kms_key_version = args.CONCEPTS.kms_key_version.Parse()
    if args.generate_key:
        private_key, public_key = key_generation.RSAKeyGen(2048)
        key_generation.ExportPrivateKey(args.key_output_file, private_key)
        return public_key
    elif kms_key_version:
        public_key_response = cryptokeyversions.GetPublicKey(kms_key_version)
        return bytes(public_key_response.pem) if six.PY2 else bytes(public_key_response.pem, 'utf-8')
    else:
        raise exceptions.OneOfArgumentsRequiredException(['--csr', '--generate-key', '--kms-key-version'], 'To create a certificate, please specify either a CSR, the --generate-key flag to create a new key, or the --kms-key-version flag to use an existing KMS key.')