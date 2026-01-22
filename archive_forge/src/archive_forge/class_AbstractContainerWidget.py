from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
class AbstractContainerWidget(CanvasWidget):
    """
    An abstract class for canvas widgets that contain a single child,
    such as ``BoxWidget`` and ``OvalWidget``.  Subclasses must define
    a constructor, which should create any new graphical elements and
    then call the ``AbstractCanvasContainer`` constructor.  Subclasses
    must also define the ``_update`` method and the ``_tags`` method;
    and any subclasses that define attributes should define
    ``__setitem__`` and ``__getitem__``.
    """

    def __init__(self, canvas, child, **attribs):
        """
        Create a new container widget.  This constructor should only
        be called by subclass constructors.

        :type canvas: Tkinter.Canvas
        :param canvas: This canvas widget's canvas.
        :param child: The container's child widget.  ``child`` must not
            have a parent.
        :type child: CanvasWidget
        :param attribs: The new canvas widget's attributes.
        """
        self._child = child
        self._add_child_widget(child)
        CanvasWidget.__init__(self, canvas, **attribs)

    def _manage(self):
        self._update(self._child)

    def child(self):
        """
        :return: The child widget contained by this container widget.
        :rtype: CanvasWidget
        """
        return self._child

    def set_child(self, child):
        """
        Change the child widget contained by this container widget.

        :param child: The new child widget.  ``child`` must not have a
            parent.
        :type child: CanvasWidget
        :rtype: None
        """
        self._remove_child_widget(self._child)
        self._add_child_widget(child)
        self._child = child
        self.update(child)

    def __repr__(self):
        name = self.__class__.__name__
        if name[-6:] == 'Widget':
            name = name[:-6]
        return f'[{name}: {self._child!r}]'