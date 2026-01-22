from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PipelineRef(_messages.Message):
    """PipelineRef can be used to refer to a specific instance of a Pipeline.

  Enums:
    ResolverValueValuesEnum: Resolver is the name of the resolver that should
      perform resolution of the referenced Tekton resource.

  Fields:
    name: Name of the Pipeline.
    params: Params contains the parameters used to identify the referenced
      Tekton resource. Example entries might include "repo" or "path" but the
      set of params ultimately depends on the chosen resolver.
    resolver: Resolver is the name of the resolver that should perform
      resolution of the referenced Tekton resource.
  """

    class ResolverValueValuesEnum(_messages.Enum):
        """Resolver is the name of the resolver that should perform resolution of
    the referenced Tekton resource.

    Values:
      RESOLVER_NAME_UNSPECIFIED: Default enum type; should not be used.
      BUNDLES: Bundles resolver. https://tekton.dev/docs/pipelines/bundle-
        resolver/
      GCB_REPO: GCB repo resolver.
      GIT: Simple Git resolver. https://tekton.dev/docs/pipelines/git-
        resolver/
    """
        RESOLVER_NAME_UNSPECIFIED = 0
        BUNDLES = 1
        GCB_REPO = 2
        GIT = 3
    name = _messages.StringField(1)
    params = _messages.MessageField('Param', 2, repeated=True)
    resolver = _messages.EnumField('ResolverValueValuesEnum', 3)