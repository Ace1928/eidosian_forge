from __future__ import annotations
import hashlib
import json
from typing import TYPE_CHECKING, Any, Final, Mapping, cast
from streamlit import config
from streamlit.proto.DeckGlJsonChart_pb2 import DeckGlJsonChart as PydeckProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.util import HASHLIB_KWARGS
class PydeckMixin:

    @gather_metrics('pydeck_chart')
    def pydeck_chart(self, pydeck_obj: Deck | None=None, use_container_width: bool=False) -> DeltaGenerator:
        """Draw a chart using the PyDeck library.

        This supports 3D maps, point clouds, and more! More info about PyDeck
        at https://deckgl.readthedocs.io/en/latest/.

        These docs are also quite useful:

        - DeckGL docs: https://github.com/uber/deck.gl/tree/master/docs
        - DeckGL JSON docs: https://github.com/uber/deck.gl/tree/master/modules/json

        When using this command, Mapbox provides the map tiles to render map
        content. Note that Mapbox is a third-party product and Streamlit accepts
        no responsibility or liability of any kind for Mapbox or for any content
        or information made available by Mapbox.

        Mapbox requires users to register and provide a token before users can
        request map tiles. Currently, Streamlit provides this token for you, but
        this could change at any time. We strongly recommend all users create and
        use their own personal Mapbox token to avoid any disruptions to their
        experience. You can do this with the ``mapbox.token`` config option. The
        use of Mapbox is governed by Mapbox's Terms of Use.

        To get a token for yourself, create an account at https://mapbox.com.
        For more info on how to set config options, see
        https://docs.streamlit.io/library/advanced-features/configuration

        Parameters
        ----------
        pydeck_obj: pydeck.Deck or None
            Object specifying the PyDeck chart to draw.
        use_container_width: bool

        Example
        -------
        Here's a chart using a HexagonLayer and a ScatterplotLayer. It uses either the
        light or dark map style, based on which Streamlit theme is currently active:

        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>> import pydeck as pdk
        >>>
        >>> chart_data = pd.DataFrame(
        ...    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        ...    columns=['lat', 'lon'])
        >>>
        >>> st.pydeck_chart(pdk.Deck(
        ...     map_style=None,
        ...     initial_view_state=pdk.ViewState(
        ...         latitude=37.76,
        ...         longitude=-122.4,
        ...         zoom=11,
        ...         pitch=50,
        ...     ),
        ...     layers=[
        ...         pdk.Layer(
        ...            'HexagonLayer',
        ...            data=chart_data,
        ...            get_position='[lon, lat]',
        ...            radius=200,
        ...            elevation_scale=4,
        ...            elevation_range=[0, 1000],
        ...            pickable=True,
        ...            extruded=True,
        ...         ),
        ...         pdk.Layer(
        ...             'ScatterplotLayer',
        ...             data=chart_data,
        ...             get_position='[lon, lat]',
        ...             get_color='[200, 30, 0, 160]',
        ...             get_radius=200,
        ...         ),
        ...     ],
        ... ))

        .. output::
           https://doc-pydeck-chart.streamlit.app/
           height: 530px

        .. note::
           To make the PyDeck chart's style consistent with Streamlit's theme,
           you can set ``map_style=None`` in the ``pydeck.Deck`` object.

        """
        pydeck_proto = PydeckProto()
        marshall(pydeck_proto, pydeck_obj, use_container_width)
        return self.dg._enqueue('deck_gl_json_chart', pydeck_proto)

    @property
    def dg(self) -> DeltaGenerator:
        """Get our DeltaGenerator."""
        return cast('DeltaGenerator', self)