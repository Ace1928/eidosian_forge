import os
import time
import pytest
import xcffib.xproto  # noqa isort:skip
from xcffib.xproto import ConfigWindow, CW, EventMask, GC  # noqa isort:skip
from . import Context, XCBSurface, cairo_version  # noqa isort:skip
@pytest.mark.xfail(cairo_version() < 11200, reason='Cairo version too low')
def test_xcb_window(xcb_conn):
    width = 10
    height = 10
    wid = create_window(xcb_conn, width, height)
    xcb_conn.core.MapWindow(wid)
    xcb_conn.flush()
    start = time.time()
    while time.time() < start + 10:
        event = xcb_conn.wait_for_event()
        if isinstance(event, xcffib.xproto.ExposeEvent):
            break
    else:
        pytest.fail('Never received ExposeEvent')
    root_visual = find_root_visual(xcb_conn)
    surface = XCBSurface(xcb_conn, wid, root_visual, width, height)
    assert surface
    ctx = Context(surface)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()
    xcb_conn.flush()
    xcb_conn.core.ConfigureWindow(wid, ConfigWindow.X | ConfigWindow.Y | ConfigWindow.Width | ConfigWindow.Height, [5, 5, width * 2, height * 2])
    xcb_conn.flush()
    start = time.time()
    while time.time() < start + 10:
        event = xcb_conn.wait_for_event()
        if isinstance(event, xcffib.xproto.ConfigureNotifyEvent):
            assert event.width == 2 * width
            assert event.height == 2 * height
            width = event.width
            height = event.height
            break
    else:
        pytest.fail('Never received ConfigureNotifyEvent')
    surface.set_size(width, height)
    ctx = Context(surface)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()
    xcb_conn.flush()
    while event:
        event = xcb_conn.poll_for_event()