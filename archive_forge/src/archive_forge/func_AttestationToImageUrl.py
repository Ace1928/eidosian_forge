import base64
import binascii
import collections
import copy
import json
from typing import List, Optional, Text
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image
from containerregistry.client.v2_2 import docker_session
from containerregistry.transform.v2_2 import metadata
from googlecloudsdk.api_lib.container.images import util
from googlecloudsdk.command_lib.container.binauthz import util as binauthz_util
from googlecloudsdk.core.exceptions import Error
import httplib2
def AttestationToImageUrl(attestation):
    """Extract the image url from a DSSE of predicate type https://binaryauthorization.googleapis.com/policy_verification/*.

  This is a helper function for mapping attestations back to their respective
  images. Do not use this for signature verification.

  Args:
    attestation: The attestation in base64 encoded string form.

  Returns:
    The image url referenced in the attestation.
  """
    deser_att = json.loads(StandardOrUrlsafeBase64Decode(attestation))
    deser_payload = json.loads(StandardOrUrlsafeBase64Decode(deser_att['payload']))
    return '{}@sha256:{}'.format(deser_payload['subject'][0]['name'], deser_payload['subject'][0]['digest']['sha256'])