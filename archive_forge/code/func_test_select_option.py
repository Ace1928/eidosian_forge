import pytest
from panel.tests.util import serve_component, wait_until
from panel.widgets import Select
def test_select_option(page):
    select = Select(value='B', options=['A', 'B', 'C'], size=4)
    serve_component(page, select)
    wait_until(lambda: page.locator('select').evaluate('(sel)=>sel.value') == 'B', page)