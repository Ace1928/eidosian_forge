from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.binauthz import apis
Evaluate a policy against a Kubernetes resource.

    Args:
      policy_ref: the resource name of the policy.
      resource: the Kubernetes resource in JSON or YAML form.
      generate_deploy_attestations: whether to sign results or not.

    Returns:
      The result of the evaluation in EvaluateGkePolicyResponse form.
    