class MouseStateHandler:
    """Simple handler that tracks the state of buttons from the mouse. If a
    button is pressed then this handler holds a True value for it.
    If the window loses focus, all buttons will be reset to False in order
    to avoid a "sticky" button state.

    For example::

        >>> win = window.Window()
        >>> mousebuttons = mouse.MouseStateHandler()
        >>> win.push_handlers(mousebuttons)

        # Hold down the "left" button...

        >>> mousebuttons[mouse.LEFT]
        True
        >>> mousebuttons[mouse.RIGHT]
        False

    """

    def __init__(self):
        self.data = {'x': 0, 'y': 0}

    def on_mouse_press(self, x, y, button, modifiers):
        self.data[button] = True

    def on_mouse_release(self, x, y, button, modifiers):
        self.data[button] = False

    def on_deactivate(self):
        self.data.clear()

    def on_mouse_motion(self, x, y, dx, dy):
        self.data['x'] = x
        self.data['y'] = y

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.data['x'] = x
        self.data['y'] = y

    def __getitem__(self, key):
        return self.data.get(key, False)