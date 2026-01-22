class MovableFrame(Frame):
    """A Frame that allows Widget repositioning.

    When a specified modifier key is held down, Widgets can be
    repositioned by dragging them. Examples of modifier keys are
    Ctrl, Alt, Shift. These are defined in the `pyglet.window.key`
    module, and start witih `MOD_`. For example::

        from pyglet.window.key import MOD_CTRL

        frame = pyglet.gui.frame.MovableFrame(mywindow, modifier=MOD_CTRL)

    For more information, see the `pyglet.window.key` submodule
    API documentation.
    """

    def __init__(self, window, order=0, modifier=0):
        super().__init__(window, order=order)
        self._modifier = modifier
        self._moving_widgets = set()

    def on_mouse_press(self, x, y, buttons, modifiers):
        if self._modifier & modifiers > 0:
            for widget in self._cells.get(self._hash(x, y), set()):
                if widget._check_hit(x, y):
                    self._moving_widgets.add(widget)
            for widget in self._moving_widgets:
                self.remove_widget(widget)
        else:
            super().on_mouse_press(x, y, buttons, modifiers)

    def on_mouse_release(self, x, y, buttons, modifiers):
        for widget in self._moving_widgets:
            self.add_widget(widget)
        self._moving_widgets.clear()
        super().on_mouse_release(x, y, buttons, modifiers)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        for widget in self._moving_widgets:
            wx, wy = widget.position
            widget.position = (wx + dx, wy + dy)
        super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)