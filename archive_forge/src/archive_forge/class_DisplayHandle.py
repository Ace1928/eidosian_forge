from binascii import b2a_hex
import os
import sys
import warnings
class DisplayHandle(object):
    """A handle on an updatable display

    Call `.update(obj)` to display a new object.

    Call `.display(obj`) to add a new instance of this display,
    and update existing instances.

    See Also
    --------

        :func:`display`, :func:`update_display`

    """

    def __init__(self, display_id=None):
        if display_id is None:
            display_id = _new_id()
        self.display_id = display_id

    def __repr__(self):
        return '<%s display_id=%s>' % (self.__class__.__name__, self.display_id)

    def display(self, obj, **kwargs):
        """Make a new display with my id, updating existing instances.

        Parameters
        ----------
        obj
            object to display
        **kwargs
            additional keyword arguments passed to display
        """
        display(obj, display_id=self.display_id, **kwargs)

    def update(self, obj, **kwargs):
        """Update existing displays with my id

        Parameters
        ----------
        obj
            object to display
        **kwargs
            additional keyword arguments passed to update_display
        """
        update_display(obj, display_id=self.display_id, **kwargs)