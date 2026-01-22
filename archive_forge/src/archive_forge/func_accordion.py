import pytest
from bokeh.models import Column as BkColumn, Div
import panel as pn
from panel.layout import Accordion
from panel.models import Card
@pytest.fixture
def accordion(document, comm):
    """Set up a accordion instance"""
    div1, div2 = (Div(), Div())
    return Accordion(('Tab1', div1), ('Tab2', div2))