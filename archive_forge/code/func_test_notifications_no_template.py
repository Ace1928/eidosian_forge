import pytest
from playwright.sync_api import expect
from panel.config import config
from panel.io.state import state
from panel.pane import Markdown
from panel.template import BootstrapTemplate
from panel.tests.util import serve_component
from panel.widgets import Button
def test_notifications_no_template(page):

    def callback(event):
        state.notifications.error('MyError')

    def app():
        config.notifications = True
        button = Button(name='Display error')
        button.on_click(callback)
        return button
    serve_component(page, app)
    page.click('.bk-btn')
    expect(page.locator('.notyf__message')).to_have_text('MyError')