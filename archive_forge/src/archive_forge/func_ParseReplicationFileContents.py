from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def ParseReplicationFileContents(file_contents):
    """Reads replication policy file contents and returns its data.

  Reads the contents of a json or yaml replication policy file which conforms to
  https://cloud.google.com/secret-manager/docs/reference/rest/v1/projects.secrets#replication
  and returns data needed to create a Secret with that policy. If the file
  doesn't conform to the expected format, a BadFileException is raised.

  For Secrets with an automtic policy, locations is empty and keys has
  either 0 or 1 entry depending on whether the policy includes CMEK. For Secrets
  with a user managed policy, the number of keys returns is either 0 or is equal
  to the number of locations returned with the Nth key corresponding to the Nth
  location.

  Args:
      file_contents (str): The unvalidated contents fo the replication file.

  Returns:
      result (str): Either "user-managed" or "automatic".
      locations (list): Locations that are part of the user-managed replication
      keys (list): list of kms keys to be used for each replica.
  """
    try:
        replication_policy = json.loads(file_contents)
        return _ParseReplicationDict(replication_policy)
    except ValueError:
        pass
    try:
        replication_policy = yaml.load(file_contents)
        return _ParseReplicationDict(replication_policy)
    except yaml.YAMLParseError:
        raise exceptions.BadFileException('Failed to parse replication policy file as json or yaml.')