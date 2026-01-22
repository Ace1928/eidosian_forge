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
def _GetOrCreateAlias(self, resource):
    """What do we call this secret within this resource?

    Note that there might be an existing alias to the same secret, which we'd
    like to reuse. There's no effort to deduplicate the ReachableSecret python
    objects; you just get the same alias from more than one of them.

    The k8s_object annotation is edited here to include all new aliases. Use
    PruneAnnotation to clean up unused ones.

    Args:
      resource: k8s_object resource that will be modified if we need to add a
        new alias to the secrets annotation.

    Returns:
      str for use as SecretVolumeSource.secret_name or SecretKeySelector.name
    """
    if not self._IsRemote():
        return self.secret_name
    formatted_annotation = _GetSecretsAnnotation(resource)
    remotes = ParseAnnotation(formatted_annotation)
    for alias, other_rs in remotes.items():
        if self == other_rs:
            return alias
    new_alias = self.secret_name[:5] + '-' + str(uuid.uuid1())
    remotes[new_alias] = self
    _SetSecretsAnnotation(resource, _FormatAnnotation(remotes))
    return new_alias