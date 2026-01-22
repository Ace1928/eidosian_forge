from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.cache import exceptions
from googlecloudsdk.core.cache import persistent_cache_base
import six
@classmethod
def Row(cls, name=None, columns=None, keys=None, timeout=None, modified=None, restricted=None, version=None):
    """Constructs and returns a metadata table row from the args."""
    if restricted is not None:
        restricted = int(restricted)
    return (name, columns, keys, timeout, modified, restricted, version)