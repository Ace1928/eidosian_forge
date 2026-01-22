from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import enum
import itertools
import re
import uuid
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import platforms
def _SetSecretsAnnotation(resource, value):
    annotations = resource.template.annotations
    if value:
        annotations[container_resource.SECRETS_ANNOTATION] = value
    else:
        try:
            del annotations[container_resource.SECRETS_ANNOTATION]
        except KeyError:
            pass