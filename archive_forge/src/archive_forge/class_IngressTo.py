from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IngressTo(_messages.Message):
    """Defines the conditions under which an IngressPolicy matches a request.
  Conditions are based on information about the ApiOperation intended to be
  performed on the target resource of the request. The request must satisfy
  what is defined in `operations` AND `resources` in order to match.

  Fields:
    operations: A list of ApiOperations the sources specified in corresponding
      IngressFrom are allowed to perform in this ServicePerimeter.
    resources: A list of resources, currently only projects in the form
      `projects/`, protected by this ServicePerimeter that are allowed to be
      accessed by sources defined in the corresponding IngressFrom. If a
      single `*` is specified, then access to all resources inside the
      perimeter are allowed.
    roles: IAM roles that represent the set of operations that the sources
      specified in the corresponding IngressFrom are allowed to perform in
      this ServicePerimeter.
  """
    operations = _messages.MessageField('ApiOperation', 1, repeated=True)
    resources = _messages.StringField(2, repeated=True)
    roles = _messages.StringField(3, repeated=True)