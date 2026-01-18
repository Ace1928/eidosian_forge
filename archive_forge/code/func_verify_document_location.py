import time
import pytest
import panel as pn
from panel.tests.util import serve_component, wait_until
from panel.util import parse_query
from panel.widgets import FloatSlider, RangeSlider, TextInput
def verify_document_location(expected_location, page):
    for param in expected_location:
        wait_until(lambda: param in page.evaluate('() => document.location'), page)
        wait_until(lambda: page.evaluate('() => document.location')[param] == expected_location[param], page)