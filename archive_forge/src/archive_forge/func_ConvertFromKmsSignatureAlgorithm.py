from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.binauthz import apis
from googlecloudsdk.api_lib.container.binauthz import util
from googlecloudsdk.command_lib.container.binauthz import exceptions
from googlecloudsdk.command_lib.kms import maps as kms_maps
def ConvertFromKmsSignatureAlgorithm(self, kms_algorithm):
    """Convert a KMS SignatureAlgorithm into a Binauthz SignatureAlgorithm."""
    binauthz_enum = self.messages.PkixPublicKey.SignatureAlgorithmValueValuesEnum
    kms_enum = kms_maps.ALGORITHM_ENUM
    alg_map = {kms_enum.RSA_SIGN_PSS_2048_SHA256.name: binauthz_enum.RSA_PSS_2048_SHA256, kms_enum.RSA_SIGN_PSS_3072_SHA256.name: binauthz_enum.RSA_PSS_3072_SHA256, kms_enum.RSA_SIGN_PSS_4096_SHA256.name: binauthz_enum.RSA_PSS_4096_SHA256, kms_enum.RSA_SIGN_PSS_4096_SHA512.name: binauthz_enum.RSA_PSS_4096_SHA512, kms_enum.RSA_SIGN_PKCS1_2048_SHA256.name: binauthz_enum.RSA_SIGN_PKCS1_2048_SHA256, kms_enum.RSA_SIGN_PKCS1_3072_SHA256.name: binauthz_enum.RSA_SIGN_PKCS1_3072_SHA256, kms_enum.RSA_SIGN_PKCS1_4096_SHA256.name: binauthz_enum.RSA_SIGN_PKCS1_4096_SHA256, kms_enum.RSA_SIGN_PKCS1_4096_SHA512.name: binauthz_enum.RSA_SIGN_PKCS1_4096_SHA512, kms_enum.EC_SIGN_P256_SHA256.name: binauthz_enum.ECDSA_P256_SHA256, kms_enum.EC_SIGN_P384_SHA384.name: binauthz_enum.ECDSA_P384_SHA384}
    try:
        return alg_map[kms_algorithm.name]
    except KeyError:
        raise exceptions.InvalidArgumentError('Unsupported PkixPublicKey signature algorithm: "{}"'.format(kms_algorithm.name))