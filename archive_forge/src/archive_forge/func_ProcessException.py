from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.functions.v1 import exceptions
from googlecloudsdk.calliope import exceptions as base_exceptions
from six.moves import http_client
def ProcessException(http_exception, kms_key=None):
    if kms_key and http_exception.status_code == http_client.INTERNAL_SERVER_ERROR:
        raise exceptions.FunctionsError('An error occurred. Ensure that the KMS key {kms_key} exists and the Cloud Functions service account has encrypter/decrypter permissions (roles/cloudkms.cryptoKeyEncrypterDecrypter) on the key. If you have recently made changes to the IAM config, wait a few minutes for the config to propagate and try again.'.format(kms_key=kms_key))