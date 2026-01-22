import pytest
from bokeh.models import Div
from playwright.sync_api import expect
from panel import Accordion
from panel.tests.util import serve_component
@pytest.fixture
def accordion_components():
    d0 = Div(name='Div 0', text='Text 0')
    d1 = Div(name='Div 1', text='Text 1')
    return (d0, d1)