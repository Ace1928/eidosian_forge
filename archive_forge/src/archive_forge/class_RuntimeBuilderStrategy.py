from source directory to docker image. They are stored as templated .yaml files
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import enum
import os
import re
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import config as cloudbuild_config
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
import six
import six.moves.urllib.error
import six.moves.urllib.parse
import six.moves.urllib.request
class RuntimeBuilderStrategy(enum.Enum):
    """Enum indicating when to use runtime builders."""
    NEVER = 1
    ALLOWLIST_BETA = 2
    ALLOWLIST_GA = 3
    ALWAYS = 4

    def _GetAllowlist(self):
        """Return the allowlist of runtimes for this strategy.

    The allowlist is kept as a constant within this module.

    Returns:
      list of str, the names of runtimes that are allowed for this strategy.

    Raises:
      ValueError: if this strategy is not allowlist-based.
    """
        if self is self.ALLOWLIST_GA:
            return _ALLOWLISTED_RUNTIMES_GA
        elif self is self.ALLOWLIST_BETA:
            return _ALLOWLISTED_RUNTIMES_BETA
        raise ValueError('RuntimeBuilderStrategy {} is not an allowed strategy.'.format(self))

    def _IsAllowed(self, runtime):
        for allowlisted_runtime in self._GetAllowlist():
            try:
                if allowlisted_runtime.match(runtime):
                    return True
            except AttributeError:
                if runtime == allowlisted_runtime:
                    return True
        return False

    def ShouldUseRuntimeBuilders(self, runtime, needs_dockerfile):
        """Returns True if runtime should use runtime builders under this strategy.

    For the most part, this is obvious: the ALWAYS strategy returns True, the
    ALLOWLIST_${TRACK} strategies return True if the given runtime is in the
    list of _ALLOWLISTED_RUNTIMES_${TRACK}, and the NEVER strategy returns
    False.

    However, in the case of 'custom' runtimes, things get tricky: if the
    strategy *is not* NEVER, we return True only if there is no `Dockerfile` in
    the current directory (this method assumes that there is *either* a
    `Dockerfile` or a `cloudbuild.yaml` file), since one needs to get generated
    by the Cloud Build.

    Args:
      runtime: str, the runtime being built.
      needs_dockerfile: bool, whether the Dockerfile in the source directory is
        absent.

    Returns:
      bool, whether to use the runtime builders.
    Raises:
      ValueError: if an unrecognized runtime_builder_strategy is given
    """
        if runtime == 'custom' and self in (self.ALWAYS, self.ALLOWLIST_BETA, self.ALLOWLIST_GA):
            return needs_dockerfile
        if self is self.ALWAYS:
            return True
        elif self is self.ALLOWLIST_BETA or self is self.ALLOWLIST_GA:
            return self._IsAllowed(runtime)
        elif self is self.NEVER:
            return False
        else:
            raise ValueError('Invalid runtime builder strategy [{}].'.format(self))