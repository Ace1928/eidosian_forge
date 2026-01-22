from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigureSearchRequest(_messages.Message):
    """Request to configure the search parameters for the specified FHIR store.

  Fields:
    canonicalUrls: The canonical URLs of the search parameters that are
      intended to be used for the FHIR store. See
      https://www.hl7.org/fhir/references.html#canonical for explanation on
      FHIR canonical urls
    validateOnly: If `validate_only` is set to true, the method will compile
      all the search parameters without actually setting the search config for
      the store and triggering the reindex.
  """
    canonicalUrls = _messages.StringField(1, repeated=True)
    validateOnly = _messages.BooleanField(2)