from __future__ import annotations
import logging # isort:skip
import sys
import types
import xyzservices
from bokeh.core.enums import enumeration
from .util.deprecation import deprecated
Use this function to retrieve an instance of a predefined tile provider.

        .. warning::
            get_provider is deprecated as of Bokeh 3.0.0 and will be removed in a future
            release. Use ``add_tile`` directly instead.

        Args:
            provider_name (Union[str, Vendors, xyzservices.TileProvider]):
                Name of the tile provider to supply.

                Use a ``tile_providers.Vendors`` enumeration value, or the string
                name of one of the known providers. Use
                :class:`xyzservices.TileProvider` to pass custom tile providers.

        Returns:
            WMTSTileProviderSource: The desired tile provider instance.

        Raises:
            ValueError: if the specified provider can not be found.

        Example:

            .. code-block:: python

                    >>> from bokeh.tile_providers import get_provider, Vendors
                    >>> get_provider(Vendors.CARTODBPOSITRON)
                    <class 'bokeh.models.tiles.WMTSTileSource'>
                    >>> get_provider('CARTODBPOSITRON')
                    <class 'bokeh.models.tiles.WMTSTileSource'>

                    >>> import xyzservices.providers as xyz
                    >>> get_provider(xyz.CartoDB.Positron)
                    <class 'bokeh.models.tiles.WMTSTileSource'>
        