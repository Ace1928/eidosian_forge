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
@pytest.mark.parametrize('sorter', ['sorter', 'no_sorter'])
@pytest.mark.parametrize('python_filter', ['python_filter', 'no_python_filter'])
@pytest.mark.parametrize('header_filter', ['header_filter', 'no_header_filter'])
@pytest.mark.parametrize('pagination', ['remote', 'local', 'no_pagination'])
def test_tabulator_edit_event_integrations(page, sorter, python_filter, header_filter, pagination):
    sorter_col = 'col3'
    python_filter_col = 'col2'
    python_filter_val = 'd'
    header_filter_col = 'col1'
    header_filter_val = 'Y'
    target_col = 'col4'
    target_val = 'G'
    new_val = 'GG'
    df = pd.DataFrame({'col1': list('XYYYYYYZ'), 'col2': list('abcddddd'), 'col3': list(range(8)), 'col4': list('ABCDEFGH')})
    target_index = df.set_index(target_col).index.get_loc(target_val)
    kwargs = {}
    if pagination != 'no_pagination':
        kwargs = dict(pagination=pagination, page_size=3)
    if header_filter == 'header_filter':
        kwargs.update(dict(header_filters={header_filter_col: {'type': 'input', 'func': 'like'}}))
    widget = Tabulator(df, **kwargs)
    if python_filter == 'python_filter':
        widget.add_filter(python_filter_val, python_filter_col)
    values = []
    widget.on_edit(lambda e: values.append((e.column, e.row, e.old, e.value)))
    serve_component(page, widget)
    if sorter == 'sorter':
        s = page.locator('.tabulator-col', has_text=sorter_col).locator('.tabulator-col-sorter')
        s.click()
        page.wait_for_timeout(200)
        s.click()
        page.wait_for_timeout(200)
    if header_filter == 'header_filter':
        str_header = page.locator('input[type="search"]')
        str_header.click()
        str_header.fill(header_filter_val)
        str_header.press('Enter')
        wait_until(lambda: len(widget.filters) == 1, page)
    if pagination != 'no_pagination' and sorter == 'no_sorter':
        page.locator('text="Last"').click()
        page.wait_for_timeout(200)
    cell = page.locator(f'text="{target_val}"')
    cell.click()
    editable_cell = page.locator('input[type="text"]')
    editable_cell.fill(new_val)
    editable_cell.press('Enter')
    wait_until(lambda: len(values) == 1, page)
    assert values[0] == (target_col, target_index, target_val, new_val)
    assert df[target_col][target_index] == new_val
    assert widget.value.equals(df)
    if sorter == 'sorter':
        expected_current_view = widget.value.sort_values(sorter_col, ascending=False)
    else:
        expected_current_view = widget.value
    if python_filter == 'python_filter':
        expected_current_view = expected_current_view.query(f'{python_filter_col} == @python_filter_val')
    if header_filter == 'header_filter':
        expected_current_view = expected_current_view.query(f'{header_filter_col} == @header_filter_val')
    assert widget.current_view.equals(expected_current_view)