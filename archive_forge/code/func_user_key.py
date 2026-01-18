from pyglet import compat_platform
def user_key(scancode):
    """Return a key symbol for a key not supported by pyglet.

    This can be used to map virtual keys or scancodes from unsupported
    keyboard layouts into a machine-specific symbol.  The symbol will
    be meaningless on any other machine, or under a different keyboard layout.

    Applications should use user-keys only when user explicitly binds them
    (for example, mapping keys to actions in a game options screen).
    """
    assert scancode > 0
    return scancode << 32