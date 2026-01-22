from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsSimulationsGetRequest(_messages.Message):
    """A SecuritycenterOrganizationsSimulationsGetRequest object.

  Fields:
    name: Required. The organization name or simulation name of this
      simulation Valid format:
      "organizations/{organization}/simulations/latest"
      "organizations/{organization}/simulations/{simulation}"
  """
    name = _messages.StringField(1, required=True)