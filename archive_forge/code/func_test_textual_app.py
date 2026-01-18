import pytest
from playwright.sync_api import expect
from panel.pane import Textual
from panel.tests.util import serve_component, wait_until
def test_textual_app(page):
    clicks = []

    def app():

        class ButtonApp(App):

            def compose(self):
                yield Button('Default')

            def on_button_pressed(self, event: Button.Pressed) -> None:
                clicks.append(event)
        app = ButtonApp()
        textual = Textual(app)
        return textual
    serve_component(page, app)
    expect(page.locator('.xterm-screen')).to_have_count(1)
    wait_until(lambda: bool(page.mouse.click(50, 50) or clicks), page)