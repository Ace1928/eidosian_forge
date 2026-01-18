import os
import time
import pytest
import xcffib.xproto  # noqa isort:skip
from xcffib.xproto import ConfigWindow, CW, EventMask, GC  # noqa isort:skip
from . import Context, XCBSurface, cairo_version  # noqa isort:skip
@pytest.mark.xfail(cairo_version() < 11200, reason='Cairo version too low')
def test_xcb_pixmap(xcb_conn):
    width = 10
    height = 10
    wid = create_window(xcb_conn, width, height)
    pixmap = create_pixmap(xcb_conn, wid, width, height)
    gc = create_gc(xcb_conn)
    root_visual = find_root_visual(xcb_conn)
    surface = XCBSurface(xcb_conn, pixmap, root_visual, width, height)
    assert surface
    ctx = Context(surface)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()
    xcb_conn.core.MapWindow(wid)
    xcb_conn.flush()
    start = time.time()
    while time.time() < start + 10:
        event = xcb_conn.wait_for_event()
        if isinstance(event, xcffib.xproto.ExposeEvent):
            break
    else:
        pytest.fail('Never received ExposeEvent')
    xcb_conn.core.CopyArea(pixmap, wid, gc, 0, 0, 0, 0, width, height)
    ctx = None
    surface = None
    xcb_conn.core.FreeGC(gc)
    xcb_conn.core.FreePixmap(pixmap)
    xcb_conn.flush()
    while event:
        event = xcb_conn.poll_for_event()