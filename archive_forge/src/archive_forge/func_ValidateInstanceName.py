from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import exceptions as sql_exceptions
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.calliope import exceptions
def ValidateInstanceName(instance_name):
    if ':' in instance_name:
        name_components = instance_name.split(':')
        possible_project = name_components[0]
        possible_instance = name_components[-1]
        raise sql_exceptions.ArgumentError("Instance names cannot contain the ':' character. If you meant to indicate the\nproject for [{instance}], use only '{instance}' for the argument, and either add\n'--project {project}' to the command line or first run\n  $ gcloud config set project {project}\n".format(project=possible_project, instance=possible_instance))