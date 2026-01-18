from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def unbind_click(self, button=1):
    """
        Remove a callback that was registered with ``bind_click``.

        :type button: int
        :param button: Which button the user should use to click on
            this ``CanvasWidget``.  Typically, this should be 1 (left
            button), 3 (right button), or 2 (middle button).
        """
    try:
        del self.__callbacks[button]
    except:
        pass