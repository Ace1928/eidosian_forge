from __future__ import annotations
import logging # isort:skip
from ..util.sampledata import load_json, package_path
 Provide medal counts by country for the 2014 Olympics.

Sourced from public news sources in 2014.

This module contains a single dict: ``data``.

The dictionary has a key ``"data"`` that lists sub-dictionaries, one for each
country:

.. code-block:: python

    {
        'abbr': 'DEU',
        'medals': {'total': 15, 'bronze': 4, 'gold': 8, 'silver': 3},
        'name': 'Germany'
    }

.. bokeh-sampledata-xref:: olympics2014
