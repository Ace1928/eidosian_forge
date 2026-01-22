from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.api_lib.firebase.test import util
def GetDefaultLocale(self):
    """Return the default iOS locale."""
    locales = self.catalog.runtimeConfiguration.locales
    locale = self._default_locale if self._default_locale else self._FindDefaultDimension(locales)
    if not locale:
        raise exceptions.DefaultDimensionNotFoundError(_LOCALE_DIMENSION)
    return locale