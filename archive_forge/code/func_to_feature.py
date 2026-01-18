from itertools import chain
import json
import re
import click
def to_feature(obj):
    """Converts an object to a GeoJSON Feature

    Returns feature verbatim or wraps geom in a feature with empty
    properties.

    Raises
    ------
    ValueError

    Returns
    -------
    Mapping
        A GeoJSON Feature represented by a Python mapping

    """
    if obj['type'] == 'Feature':
        return obj
    elif 'coordinates' in obj:
        return {'type': 'Feature', 'properties': {}, 'geometry': obj}
    else:
        raise ValueError('Object is not a feature or geometry')