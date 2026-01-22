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
def _CreateSbomRefNoteIfNotExists(project_id, sbom):
    """Create the SBOM reference note if not exists.

  Args:
    project_id: str, the project we will use to create the note.
    sbom: SbomFile, metadata of the SBOM file.

  Returns:
    A Note object for the targeting SBOM reference note.
  """
    client = ca_requests.GetClient()
    messages = ca_requests.GetMessages()
    note_id = _GetReferenceNoteID(sbom.sbom_format, sbom.version)
    name = resources.REGISTRY.Create(collection='containeranalysis.projects.notes', projectsId=project_id, notesId=note_id).RelativeName()
    try:
        get_request = messages.ContaineranalysisProjectsNotesGetRequest(name=name)
        note = client.projects_notes.Get(get_request)
    except apitools_exceptions.HttpNotFoundError:
        log.debug('Note not found. Creating note {0}.'.format(name))
        sbom_reference = messages.SBOMReferenceNote(format=sbom.sbom_format, version=sbom.version)
        new_note = messages.Note(kind=messages.Note.KindValueValuesEnum.SBOM_REFERENCE, sbomReference=sbom_reference)
        create_request = messages.ContaineranalysisProjectsNotesCreateRequest(parent='projects/{project}'.format(project=project_id), noteId=note_id, note=new_note)
        note = client.projects_notes.Create(create_request)
    log.debug('get note results: {0}'.format(note))
    return note