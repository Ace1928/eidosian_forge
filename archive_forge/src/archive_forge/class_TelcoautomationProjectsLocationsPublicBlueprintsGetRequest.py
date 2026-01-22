from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsPublicBlueprintsGetRequest(_messages.Message):
    """A TelcoautomationProjectsLocationsPublicBlueprintsGetRequest object.

  Fields:
    name: Required. The name of the public blueprint.
  """
    name = _messages.StringField(1, required=True)