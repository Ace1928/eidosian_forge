from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
ValueValueValuesEnum enum type.

        Values:
          RACK_TYPE_UNSPECIFIED: Unspecified rack type, single rack also
            belongs to this type.
          BASE: Base rack type, a pair of two modified Config-1 racks
            containing Aggregation switches.
          EXPANSION: Expansion rack type, also known as standalone racks,
            added by customers on demand.
        