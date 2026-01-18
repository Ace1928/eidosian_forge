import pytest
from panel.config import panel_extension as extension
from panel.pane import Markdown
from panel.tests.util import serve_component
def test_serve_page_with_reactive_html_css_on_nested_route(page):

    def app():
        extension(notifications=True, template='material')
        Markdown('Initial').servable()
    msgs, _ = serve_component(page, {'/foo/bar': app})
    assert [msg for msg in msgs if msg.type == 'error'] == []