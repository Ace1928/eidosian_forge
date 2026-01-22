from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
import json
import random
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_http as v2_2_docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.client.v2_2 import docker_image_list as v2_2_image_list
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.api_lib.container.images import util as gcr_util
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.artifacts import util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import transports
from googlecloudsdk.core.util import files
import requests
import six
from six.moves import urllib
def _VerifyGCSObject(occ):
    """Verify the existence and the content of a GCS SBOM file object.

  Args:
    occ: SBOM reference occurrence.

  Returns:
    An SbomReference object with the input occurrence and SBOM file information.
  """
    gcs_client = storage_api.StorageClient()
    obj_ref = storage_util.ObjectReference.FromUrl(occ.sbomReference.payload.predicate.location)
    file_info = {}
    try:
        gcs_client.GetObject(obj_ref)
    except apitools_exceptions.HttpNotFoundError:
        file_info['exists'] = False
    except apitools_exceptions.HttpError as e:
        msg = json.loads(e.content)
        file_info['err_msg'] = msg['error']['message']
    except Exception as e:
        file_info['err_msg'] = str(e)
    else:
        file_info['exists'] = True
    return SbomReference(occ, file_info)