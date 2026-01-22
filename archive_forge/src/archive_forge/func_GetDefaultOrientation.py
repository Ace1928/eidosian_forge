from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.api_lib.firebase.test import util
def GetDefaultOrientation(self):
    """Return the default iOS orientation."""
    orientations = self.catalog.runtimeConfiguration.orientations
    orientation = self._default_orientation if self._default_orientation else self._FindDefaultDimension(orientations)
    if not orientation:
        raise exceptions.DefaultDimensionNotFoundError(_ORIENTATION_DIMENSION)
    return orientation