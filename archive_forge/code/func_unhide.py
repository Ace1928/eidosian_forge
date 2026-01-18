from pyglet.libs.darwin import cocoapy
@classmethod
def unhide(cls):
    if cls.cursor_is_hidden:
        cocoapy.send_message('NSCursor', 'unhide')
        cls.cursor_is_hidden = False