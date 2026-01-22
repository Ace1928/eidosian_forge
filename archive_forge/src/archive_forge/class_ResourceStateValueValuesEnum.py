from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceStateValueValuesEnum(_messages.Enum):
    """The state of this domain as a `Registration` resource.

    Values:
      RESOURCE_STATE_UNSPECIFIED: The assessment is undefined.
      IMPORTABLE: A `Registration` resource can be created for this domain by
        calling `ImportDomain`.
      UNSUPPORTED: A `Registration` resource cannot be created for this domain
        because it is not supported by Cloud Domains; for example, the top-
        level domain is not supported or the registry charges non-standard
        pricing for yearly renewals.
      SUSPENDED: A `Registration` resource cannot be created for this domain
        because it is suspended and needs to be resolved with Google Domains.
      EXPIRED: A `Registration` resource cannot be created for this domain
        because it is expired and needs to be renewed with Google Domains.
      DELETED: A `Registration` resource cannot be created for this domain
        because it is deleted, but it may be possible to restore it with
        Google Domains.
    """
    RESOURCE_STATE_UNSPECIFIED = 0
    IMPORTABLE = 1
    UNSUPPORTED = 2
    SUSPENDED = 3
    EXPIRED = 4
    DELETED = 5