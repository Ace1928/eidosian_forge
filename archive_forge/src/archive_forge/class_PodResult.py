from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PodResult(_messages.Message):
    """Result of evaluating the whole GKE policy for one Pod.

  Enums:
    VerdictValueValuesEnum: The result of evaluating this Pod.

  Fields:
    imageResults: Per-image details.
    kubernetesNamespace: The Kubernetes namespace of the Pod.
    kubernetesServiceAccount: The Kubernetes service account of the Pod.
    podName: The name of the Pod.
    verdict: The result of evaluating this Pod.
  """

    class VerdictValueValuesEnum(_messages.Enum):
        """The result of evaluating this Pod.

    Values:
      POD_VERDICT_UNSPECIFIED: Not specified. This should never be used.
      CONFORMANT: All images conform to the policy.
      NON_CONFORMANT: At least one image does not conform to the policy.
      ERROR: Encountered at least one error evaluating an image and all other
        images with non-error verdicts conform to the policy. Non-conformance
        has precedence over errors.
    """
        POD_VERDICT_UNSPECIFIED = 0
        CONFORMANT = 1
        NON_CONFORMANT = 2
        ERROR = 3
    imageResults = _messages.MessageField('ImageResult', 1, repeated=True)
    kubernetesNamespace = _messages.StringField(2)
    kubernetesServiceAccount = _messages.StringField(3)
    podName = _messages.StringField(4)
    verdict = _messages.EnumField('VerdictValueValuesEnum', 5)