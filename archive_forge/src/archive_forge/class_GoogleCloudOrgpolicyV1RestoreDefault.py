from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudOrgpolicyV1RestoreDefault(_messages.Message):
    """Ignores policies set above this resource and restores the
  `constraint_default` enforcement behavior of the specific `Constraint` at
  this resource. Suppose that `constraint_default` is set to `ALLOW` for the
  `Constraint` `constraints/serviceuser.services`. Suppose that organization
  foo.com sets a `Policy` at their Organization resource node that restricts
  the allowed service activations to deny all service activations. They could
  then set a `Policy` with the `policy_type` `restore_default` on several
  experimental projects, restoring the `constraint_default` enforcement of the
  `Constraint` for only those projects, allowing those projects to have all
  services activated.
  """