from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CustomResourceDefinitionVersion(_messages.Message):
    """A CustomResourceDefinitionVersion object.

  Fields:
    additionalPrinterColumns: additionalPrinterColumns specifies additional
      columns returned in Table output. See
      https://kubernetes.io/docs/reference/using-api/api-concepts/#receiving-
      resources-as-tables for details. If no columns are specified, a single
      column displaying the age of the custom resource is used.
    deprecated: deprecated indicates this version of the custom resource API
      is deprecated. When set to true, API requests to this version receive a
      warning header in the server response. Defaults to false.
    deprecationWarning: deprecationWarning overrides the default warning
      returned to API clients. May only be set when `deprecated` is true. The
      default warning indicates this version is deprecated and recommends use
      of the newest served version of equal or greater stability, if one
      exists.
    name: Name is the version name, e.g. "v1", "v2beta1", etc.
    schema: schema describes the schema used for validation, pruning, and
      defaulting of this version of the custom resource.
    served: Served is a flag enabling/disabling this version from being served
      via REST APIs
    storage: Storage flags the version as storage version. There must be
      exactly one flagged as storage version.
    subresources: subresources specify what subresources this version of the
      defined custom resource have.
  """
    additionalPrinterColumns = _messages.MessageField('CustomResourceColumnDefinition', 1, repeated=True)
    deprecated = _messages.BooleanField(2)
    deprecationWarning = _messages.StringField(3)
    name = _messages.StringField(4)
    schema = _messages.MessageField('CustomResourceValidation', 5)
    served = _messages.BooleanField(6)
    storage = _messages.BooleanField(7)
    subresources = _messages.MessageField('CustomResourceSubresources', 8)