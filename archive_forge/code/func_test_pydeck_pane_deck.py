import numpy as np
import pytest
from bokeh.core.serialization import Serializer
from panel.models.deckgl import DeckGLPlot
from panel.pane import DeckGL, PaneBase, panel
@pydeck_available
def test_pydeck_pane_deck(document, comm):
    deck = pydeck.Deck(tooltip=True, api_keys={'mapbox': 'ABC'})
    pane = panel(deck)
    model = pane.get_root(document, comm=comm)
    assert isinstance(model, DeckGLPlot)
    assert pane._models[model.ref['id']][0] is model
    expected = {'mapProvider': 'carto', 'mapStyle': 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json', 'views': [{'@@type': 'MapView', 'controller': True}]}
    if 'tooltip' in model.data:
        expected['tooltip'] = True
    assert model.data == expected
    assert model.mapbox_api_key == deck.mapbox_key
    assert model.tooltip == deck.deck_widget.tooltip
    new_deck = pydeck.Deck(tooltip=False)
    pane.object = new_deck
    assert pane._models[model.ref['id']][0] is model
    assert model.tooltip == new_deck.deck_widget.tooltip
    pane._cleanup(model)
    assert pane._models == {}