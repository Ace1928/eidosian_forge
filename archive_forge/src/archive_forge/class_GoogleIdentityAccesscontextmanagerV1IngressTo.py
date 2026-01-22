from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIdentityAccesscontextmanagerV1IngressTo(_messages.Message):
    """Defines the conditions under which an IngressPolicy matches a request.
  Conditions are based on information about the ApiOperation intended to be
  performed on the target resource of the request. The request must satisfy
  what is defined in `operations` AND `resources` in order to match.

  Fields:
    operations: A list of ApiOperations allowed to be performed by the sources
      specified in corresponding IngressFrom in this ServicePerimeter.
    resources: A list of resources, currently only projects in the form
      `projects/`, protected by this ServicePerimeter that are allowed to be
      accessed by sources defined in the corresponding IngressFrom. If a
      single `*` is specified, then access to all resources inside the
      perimeter are allowed.
  """
    operations = _messages.MessageField('GoogleIdentityAccesscontextmanagerV1ApiOperation', 1, repeated=True)
    resources = _messages.StringField(2, repeated=True)