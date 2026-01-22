from pyglet.libs.darwin import cocoapy
class SystemCursor:
    cursor_is_hidden = False

    @classmethod
    def hide(cls):
        if not cls.cursor_is_hidden:
            cocoapy.send_message('NSCursor', 'hide')
            cls.cursor_is_hidden = True

    @classmethod
    def unhide(cls):
        if cls.cursor_is_hidden:
            cocoapy.send_message('NSCursor', 'unhide')
            cls.cursor_is_hidden = False