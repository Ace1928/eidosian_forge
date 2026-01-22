from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import logging
import traceback
from apitools.base.py import exceptions as apitools_exceptions
from boto import config
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import BadRequestException
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import ServiceException
from gslib.gcs_json_credentials import SetUpJsonCredentialsAndCache
from gslib.no_op_credentials import NoOpCredentials
from gslib.third_party.kms_apitools import cloudkms_v1_client as apitools_client
from gslib.third_party.kms_apitools import cloudkms_v1_messages as apitools_messages
from gslib.utils import system_util
from gslib.utils.boto_util import GetCertsFile
from gslib.utils.boto_util import GetMaxRetryDelay
from gslib.utils.boto_util import GetNewHttp
from gslib.utils.boto_util import GetNumRetries
def CreateCryptoKey(self, keyring_fqn, key_name):
    """Attempts to create the specified cryptoKey.

    Args:
      keyring_fqn: (str) The fully-qualified name of the keyRing, e.g.
          projects/my-project/locations/global/keyRings/my-keyring.
      key_name: (str) The name of the desired key, e.g. my-key. Note that
          this must be unique within the keyRing.

    Returns:
      (str) The fully-qualified name of the cryptoKey, e.g.:
      projects/my-project/locations/global/keyRings/my-keyring/cryptoKeys/my-key

    Raises:
      Translated CloudApi exception if we were unable to create the cryptoKey.
      Note that in the event of a 409 status code (resource already exists) when
      attempting creation, we continue and treat this as a success.
    """
    cryptokey_msg = apitools_messages.CryptoKey(purpose=apitools_messages.CryptoKey.PurposeValueValuesEnum.ENCRYPT_DECRYPT)
    cryptokey_create_request = apitools_messages.CloudkmsProjectsLocationsKeyRingsCryptoKeysCreateRequest(cryptoKey=cryptokey_msg, cryptoKeyId=key_name, parent=keyring_fqn)
    try:
        self.api_client.projects_locations_keyRings_cryptoKeys.Create(cryptokey_create_request)
    except TRANSLATABLE_APITOOLS_EXCEPTIONS as e:
        if e.status_code != 409:
            raise
    return '%s/cryptoKeys/%s' % (keyring_fqn.rstrip('/'), key_name)