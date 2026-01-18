import warnings
Attempt to get a style URI by map provider, otherwise pass the map identifier
    to the API service

    Provide reasonable cross-provider default map styles

    Parameters
    ----------
    map_identifier : str
        Either a specific map provider style or a token indicating a map style. Currently
        tokens are "dark", "light", "satellite", "road", "dark_no_labels", or "light_no_labels".
        Not all map styles are available for all providers.
    provider : str
        One of "carto", "mapbox", or "google_maps", indicating the associated base map tile provider.

    Returns
    -------
    str
        Base map URI

    