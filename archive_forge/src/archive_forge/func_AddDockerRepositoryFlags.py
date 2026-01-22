from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from argcomplete.completers import DirectoriesCompleter
from googlecloudsdk.api_lib.functions.v1 import util as api_util
from googlecloudsdk.api_lib.functions.v2 import client as client_v2
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.eventarc import flags as eventarc_flags
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
import six
def AddDockerRepositoryFlags(parser):
    """Adds flags for configuring the Docker repository for Cloud Function."""
    kmskey_group = parser.add_group(mutex=True)
    kmskey_group.add_argument('--docker-repository', type=arg_parsers.RegexpValidator(_DOCKER_REPOSITORY_NAME_PATTERN, _DOCKER_REPOSITORY_NAME_ERROR), help="        Sets the Docker repository to be used for storing the Cloud Function's\n        Docker images while the function is being deployed. `DOCKER_REPOSITORY`\n        must be an Artifact Registry Docker repository present in the `same`\n        project and location as the Cloud Function.\n\n        The repository name should match one of these patterns:\n\n        * `projects/${PROJECT}/locations/${LOCATION}/repositories/${REPOSITORY}`,\n        * `{LOCATION}-docker.pkg.dev/{PROJECT}/{REPOSITORY}`.\n\n        where `${PROJECT}` is the project, `${LOCATION}` is the location of the\n        repository and `${REPOSITORY}` is a valid repository ID.\n      ")
    kmskey_group.add_argument('--clear-docker-repository', action='store_true', help='        Clears the Docker repository configuration of the function.\n      ')