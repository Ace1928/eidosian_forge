from __future__ import unicode_literals
class MouseEvent(object):
    """
    Mouse event, sent to `UIControl.mouse_handler`.

    :param position: `Point` instance.
    :param event_type: `MouseEventType`.
    """

    def __init__(self, position, event_type):
        self.position = position
        self.event_type = event_type

    def __repr__(self):
        return 'MouseEvent(%r, %r)' % (self.position, self.event_type)