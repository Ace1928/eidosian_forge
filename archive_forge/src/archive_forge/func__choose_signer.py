import base64
import datetime
import json
import weakref
import botocore
import botocore.auth
from botocore.awsrequest import create_request_object, prepare_request_dict
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import ArnParser, datetime2timestamp
from botocore.utils import fix_s3_host  # noqa
def _choose_signer(self, operation_name, signing_type, context):
    """
        Allow setting the signature version via the choose-signer event.
        A value of `botocore.UNSIGNED` means no signing will be performed.

        :param operation_name: The operation to sign.
        :param signing_type: The type of signing that the signer is to be used
            for.
        :return: The signature version to sign with.
        """
    signing_type_suffix_map = {'presign-post': '-presign-post', 'presign-url': '-query'}
    suffix = signing_type_suffix_map.get(signing_type, '')
    signature_version = context.get('auth_type') or self._signature_version
    signing = context.get('signing', {})
    signing_name = signing.get('signing_name', self._signing_name)
    region_name = signing.get('region', self._region_name)
    if signature_version is not botocore.UNSIGNED and (not signature_version.endswith(suffix)):
        signature_version += suffix
    handler, response = self._event_emitter.emit_until_response('choose-signer.{}.{}'.format(self._service_id.hyphenize(), operation_name), signing_name=signing_name, region_name=region_name, signature_version=signature_version, context=context)
    if response is not None:
        signature_version = response
        if signature_version is not botocore.UNSIGNED and (not signature_version.endswith(suffix)):
            signature_version += suffix
    return signature_version