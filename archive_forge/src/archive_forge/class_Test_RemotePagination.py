from __future__ import annotations
import datetime as dt
from contextlib import contextmanager
import numpy as np
import pandas as pd
import param
import pytest
from bokeh.models.widgets.tables import (
from playwright.sync_api import expect
from panel.depends import bind
from panel.io.state import state
from panel.layout.base import Column
from panel.models.tabulator import _TABULATOR_THEMES_MAPPING
from panel.tests.util import get_ctrl_modifier, serve_component, wait_until
from panel.widgets import Select, Tabulator
class Test_RemotePagination:

    @pytest.fixture(autouse=True)
    def setup_widget(self, page):
        self.widget = Tabulator(value=pd.DataFrame(np.arange(20) + 100), disabled=True, pagination='remote', page_size=10, selectable=self.selectable, header_filters=True)
        serve_component(page, self.widget)

    def check_selected(self, page, expected, ui_count=None):
        if ui_count is None:
            ui_count = len(expected)
        expect(page.locator('.tabulator-selected')).to_have_count(ui_count)
        wait_until(lambda: self.widget.selection == expected, page)

    @contextmanager
    def hold_down_ctrl(self, page):
        key = get_ctrl_modifier()
        page.keyboard.down(key)
        yield
        page.keyboard.up(key)

    @contextmanager
    def hold_down_shift(self, page):
        key = 'Shift'
        page.keyboard.down(key)
        yield
        page.keyboard.up(key)

    def get_rows(self, page):
        return page.locator('.tabulator-row[role="row"]')

    def goto_page(self, page, page_number):
        page.locator(f'button.tabulator-page[data-page="{page_number}"]').click()
        page.wait_for_timeout(100)

    def click_sorting(self, page):
        page.locator('div.tabulator-col-title').get_by_text('index').click()
        page.wait_for_timeout(100)

    def set_filtering(self, page, number):
        number_input = page.locator('input[type="number"]').first
        number_input.fill(str(number))
        number_input.press('Enter')