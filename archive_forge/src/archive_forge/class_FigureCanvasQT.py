import functools
import os
import sys
import traceback
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
import matplotlib.backends.qt_editor.figureoptions as figureoptions
from . import qt_compat
from .qt_compat import (
class FigureCanvasQT(FigureCanvasBase, QtWidgets.QWidget):
    required_interactive_framework = 'qt'
    _timer_cls = TimerQT
    manager_class = _api.classproperty(lambda cls: FigureManagerQT)
    buttond = {getattr(QtCore.Qt.MouseButton, k): v for k, v in [('LeftButton', MouseButton.LEFT), ('RightButton', MouseButton.RIGHT), ('MiddleButton', MouseButton.MIDDLE), ('XButton1', MouseButton.BACK), ('XButton2', MouseButton.FORWARD)]}

    def __init__(self, figure=None):
        _create_qApp()
        super().__init__(figure=figure)
        self._draw_pending = False
        self._is_drawing = False
        self._draw_rect_callback = lambda painter: None
        self._in_resize_event = False
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setMouseTracking(True)
        self.resize(*self.get_width_height())
        palette = QtGui.QPalette(QtGui.QColor('white'))
        self.setPalette(palette)

    def _update_pixel_ratio(self):
        if self._set_device_pixel_ratio(self.devicePixelRatioF() or 1):
            event = QtGui.QResizeEvent(self.size(), self.size())
            self.resizeEvent(event)

    def _update_screen(self, screen):
        self._update_pixel_ratio()
        if screen is not None:
            screen.physicalDotsPerInchChanged.connect(self._update_pixel_ratio)
            screen.logicalDotsPerInchChanged.connect(self._update_pixel_ratio)

    def showEvent(self, event):
        window = self.window().windowHandle()
        window.screenChanged.connect(self._update_screen)
        self._update_screen(window.screen())

    def set_cursor(self, cursor):
        self.setCursor(_api.check_getitem(cursord, cursor=cursor))

    def mouseEventCoords(self, pos=None):
        """
        Calculate mouse coordinates in physical pixels.

        Qt uses logical pixels, but the figure is scaled to physical
        pixels for rendering.  Transform to physical pixels so that
        all of the down-stream transforms work as expected.

        Also, the origin is different and needs to be corrected.
        """
        if pos is None:
            pos = self.mapFromGlobal(QtGui.QCursor.pos())
        elif hasattr(pos, 'position'):
            pos = pos.position()
        elif hasattr(pos, 'pos'):
            pos = pos.pos()
        x = pos.x()
        y = self.figure.bbox.height / self.device_pixel_ratio - pos.y()
        return (x * self.device_pixel_ratio, y * self.device_pixel_ratio)

    def enterEvent(self, event):
        mods = QtWidgets.QApplication.instance().queryKeyboardModifiers()
        LocationEvent('figure_enter_event', self, *self.mouseEventCoords(event), modifiers=self._mpl_modifiers(mods), guiEvent=event)._process()

    def leaveEvent(self, event):
        QtWidgets.QApplication.restoreOverrideCursor()
        LocationEvent('figure_leave_event', self, *self.mouseEventCoords(), modifiers=self._mpl_modifiers(), guiEvent=event)._process()

    def mousePressEvent(self, event):
        button = self.buttond.get(event.button())
        if button is not None:
            MouseEvent('button_press_event', self, *self.mouseEventCoords(event), button, modifiers=self._mpl_modifiers(), guiEvent=event)._process()

    def mouseDoubleClickEvent(self, event):
        button = self.buttond.get(event.button())
        if button is not None:
            MouseEvent('button_press_event', self, *self.mouseEventCoords(event), button, dblclick=True, modifiers=self._mpl_modifiers(), guiEvent=event)._process()

    def mouseMoveEvent(self, event):
        MouseEvent('motion_notify_event', self, *self.mouseEventCoords(event), modifiers=self._mpl_modifiers(), guiEvent=event)._process()

    def mouseReleaseEvent(self, event):
        button = self.buttond.get(event.button())
        if button is not None:
            MouseEvent('button_release_event', self, *self.mouseEventCoords(event), button, modifiers=self._mpl_modifiers(), guiEvent=event)._process()

    def wheelEvent(self, event):
        if event.pixelDelta().isNull() or QtWidgets.QApplication.instance().platformName() == 'xcb':
            steps = event.angleDelta().y() / 120
        else:
            steps = event.pixelDelta().y()
        if steps:
            MouseEvent('scroll_event', self, *self.mouseEventCoords(event), step=steps, modifiers=self._mpl_modifiers(), guiEvent=event)._process()

    def keyPressEvent(self, event):
        key = self._get_key(event)
        if key is not None:
            KeyEvent('key_press_event', self, key, *self.mouseEventCoords(), guiEvent=event)._process()

    def keyReleaseEvent(self, event):
        key = self._get_key(event)
        if key is not None:
            KeyEvent('key_release_event', self, key, *self.mouseEventCoords(), guiEvent=event)._process()

    def resizeEvent(self, event):
        if self._in_resize_event:
            return
        self._in_resize_event = True
        try:
            w = event.size().width() * self.device_pixel_ratio
            h = event.size().height() * self.device_pixel_ratio
            dpival = self.figure.dpi
            winch = w / dpival
            hinch = h / dpival
            self.figure.set_size_inches(winch, hinch, forward=False)
            QtWidgets.QWidget.resizeEvent(self, event)
            ResizeEvent('resize_event', self)._process()
            self.draw_idle()
        finally:
            self._in_resize_event = False

    def sizeHint(self):
        w, h = self.get_width_height()
        return QtCore.QSize(w, h)

    def minumumSizeHint(self):
        return QtCore.QSize(10, 10)

    @staticmethod
    def _mpl_modifiers(modifiers=None, *, exclude=None):
        if modifiers is None:
            modifiers = QtWidgets.QApplication.instance().keyboardModifiers()
        modifiers = _to_int(modifiers)
        return [SPECIAL_KEYS[key].replace('control', 'ctrl') for mask, key in _MODIFIER_KEYS if exclude != key and modifiers & mask]

    def _get_key(self, event):
        event_key = event.key()
        mods = self._mpl_modifiers(exclude=event_key)
        try:
            key = SPECIAL_KEYS[event_key]
        except KeyError:
            if event_key > sys.maxunicode:
                return None
            key = chr(event_key)
            if 'shift' in mods:
                mods.remove('shift')
            else:
                key = key.lower()
        return '+'.join(mods + [key])

    def flush_events(self):
        QtWidgets.QApplication.instance().processEvents()

    def start_event_loop(self, timeout=0):
        if hasattr(self, '_event_loop') and self._event_loop.isRunning():
            raise RuntimeError('Event loop already running')
        self._event_loop = event_loop = QtCore.QEventLoop()
        if timeout > 0:
            _ = QtCore.QTimer.singleShot(int(timeout * 1000), event_loop.quit)
        with _maybe_allow_interrupt(event_loop):
            qt_compat._exec(event_loop)

    def stop_event_loop(self, event=None):
        if hasattr(self, '_event_loop'):
            self._event_loop.quit()

    def draw(self):
        """Render the figure, and queue a request for a Qt draw."""
        if self._is_drawing:
            return
        with cbook._setattr_cm(self, _is_drawing=True):
            super().draw()
        self.update()

    def draw_idle(self):
        """Queue redraw of the Agg buffer and request Qt paintEvent."""
        if not (getattr(self, '_draw_pending', False) or getattr(self, '_is_drawing', False)):
            self._draw_pending = True
            QtCore.QTimer.singleShot(0, self._draw_idle)

    def blit(self, bbox=None):
        if bbox is None and self.figure:
            bbox = self.figure.bbox
        l, b, w, h = [int(pt / self.device_pixel_ratio) for pt in bbox.bounds]
        t = b + h
        self.repaint(l, self.rect().height() - t, w, h)

    def _draw_idle(self):
        with self._idle_draw_cntx():
            if not self._draw_pending:
                return
            self._draw_pending = False
            if self.height() < 0 or self.width() < 0:
                return
            try:
                self.draw()
            except Exception:
                traceback.print_exc()

    def drawRectangle(self, rect):
        if rect is not None:
            x0, y0, w, h = [int(pt / self.device_pixel_ratio) for pt in rect]
            x1 = x0 + w
            y1 = y0 + h

            def _draw_rect_callback(painter):
                pen = QtGui.QPen(QtGui.QColor('black'), 1 / self.device_pixel_ratio)
                pen.setDashPattern([3, 3])
                for color, offset in [(QtGui.QColor('black'), 0), (QtGui.QColor('white'), 3)]:
                    pen.setDashOffset(offset)
                    pen.setColor(color)
                    painter.setPen(pen)
                    painter.drawLine(x0, y0, x0, y1)
                    painter.drawLine(x0, y0, x1, y0)
                    painter.drawLine(x0, y1, x1, y1)
                    painter.drawLine(x1, y0, x1, y1)
        else:

            def _draw_rect_callback(painter):
                return
        self._draw_rect_callback = _draw_rect_callback
        self.update()