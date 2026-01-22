import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
class TextBuffer(Gtk.TextBuffer):

    def create_tag(self, tag_name=None, **properties):
        """Creates a tag and adds it to the tag table of the TextBuffer.

        :param str tag_name:
            Name of the new tag, or None
        :param **properties:
            Keyword list of properties and their values

        This is equivalent to creating a Gtk.TextTag and then adding the
        tag to the buffer's tag table. The returned tag is owned by
        the buffer's tag table.

        If ``tag_name`` is None, the tag is anonymous.

        If ``tag_name`` is not None, a tag called ``tag_name`` must not already
        exist in the tag table for this buffer.

        Properties are passed as a keyword list of names and values (e.g.
        foreground='DodgerBlue', weight=Pango.Weight.BOLD)

        :returns:
            A new tag.
        """
        tag = Gtk.TextTag(name=tag_name, **properties)
        self.get_tag_table().add(tag)
        return tag

    def create_mark(self, mark_name, where, left_gravity=False):
        return Gtk.TextBuffer.create_mark(self, mark_name, where, left_gravity)

    def set_text(self, text, length=-1):
        Gtk.TextBuffer.set_text(self, text, length)

    def insert(self, iter, text, length=-1):
        if not isinstance(text, str):
            raise TypeError('text must be a string, not %s' % type(text))
        Gtk.TextBuffer.insert(self, iter, text, length)

    def insert_with_tags(self, iter, text, *tags):
        start_offset = iter.get_offset()
        self.insert(iter, text)
        if not tags:
            return
        start = self.get_iter_at_offset(start_offset)
        for tag in tags:
            self.apply_tag(tag, start, iter)

    def insert_with_tags_by_name(self, iter, text, *tags):
        tag_objs = []
        for tag in tags:
            tag_obj = self.get_tag_table().lookup(tag)
            if not tag_obj:
                raise ValueError('unknown text tag: %s' % tag)
            tag_objs.append(tag_obj)
        self.insert_with_tags(iter, text, *tag_objs)

    def insert_at_cursor(self, text, length=-1):
        if not isinstance(text, str):
            raise TypeError('text must be a string, not %s' % type(text))
        Gtk.TextBuffer.insert_at_cursor(self, text, length)
    get_selection_bounds = strip_boolean_result(Gtk.TextBuffer.get_selection_bounds, fail_ret=())