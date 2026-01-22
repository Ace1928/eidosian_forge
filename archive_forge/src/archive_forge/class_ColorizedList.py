from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
class ColorizedList:
    """
    An abstract base class for displaying a colorized list of items.
    Subclasses should define:

    - ``_init_colortags``, which sets up Text color tags that
      will be used by the list.
    - ``_item_repr``, which returns a list of (text,colortag)
      tuples that make up the colorized representation of the
      item.

    :note: Typically, you will want to register a callback for
        ``'select'`` that calls ``mark`` on the given item.
    """

    def __init__(self, parent, items=[], **options):
        """
        Construct a new list.

        :param parent: The Tk widget that contains the colorized list
        :param items: The initial contents of the colorized list.
        :param options:
        """
        self._parent = parent
        self._callbacks = {}
        self._marks = {}
        self._init_itemframe(options.copy())
        self._textwidget.bind('<KeyPress>', self._keypress)
        self._textwidget.bind('<ButtonPress>', self._buttonpress)
        self._items = None
        self.set(items)

    @abstractmethod
    def _init_colortags(self, textwidget, options):
        """
        Set up any colortags that will be used by this colorized list.
        E.g.:
            textwidget.tag_config('terminal', foreground='black')
        """

    @abstractmethod
    def _item_repr(self, item):
        """
        Return a list of (text, colortag) tuples that make up the
        colorized representation of the item.  Colorized
        representations may not span multiple lines.  I.e., the text
        strings returned may not contain newline characters.
        """

    def get(self, index=None):
        """
        :return: A list of the items contained by this list.
        """
        if index is None:
            return self._items[:]
        else:
            return self._items[index]

    def set(self, items):
        """
        Modify the list of items contained by this list.
        """
        items = list(items)
        if self._items == items:
            return
        self._items = list(items)
        self._textwidget['state'] = 'normal'
        self._textwidget.delete('1.0', 'end')
        for item in items:
            for text, colortag in self._item_repr(item):
                assert '\n' not in text, 'item repr may not contain newline'
                self._textwidget.insert('end', text, colortag)
            self._textwidget.insert('end', '\n')
        self._textwidget.delete('end-1char', 'end')
        self._textwidget.mark_set('insert', '1.0')
        self._textwidget['state'] = 'disabled'
        self._marks.clear()

    def unmark(self, item=None):
        """
        Remove highlighting from the given item; or from every item,
        if no item is given.
        :raise ValueError: If ``item`` is not contained in the list.
        :raise KeyError: If ``item`` is not marked.
        """
        if item is None:
            self._marks.clear()
            self._textwidget.tag_remove('highlight', '1.0', 'end+1char')
        else:
            index = self._items.index(item)
            del self._marks[item]
            start, end = ('%d.0' % (index + 1), '%d.0' % (index + 2))
            self._textwidget.tag_remove('highlight', start, end)

    def mark(self, item):
        """
        Highlight the given item.
        :raise ValueError: If ``item`` is not contained in the list.
        """
        self._marks[item] = 1
        index = self._items.index(item)
        start, end = ('%d.0' % (index + 1), '%d.0' % (index + 2))
        self._textwidget.tag_add('highlight', start, end)

    def markonly(self, item):
        """
        Remove any current highlighting, and mark the given item.
        :raise ValueError: If ``item`` is not contained in the list.
        """
        self.unmark()
        self.mark(item)

    def view(self, item):
        """
        Adjust the view such that the given item is visible.  If
        the item is already visible, then do nothing.
        """
        index = self._items.index(item)
        self._textwidget.see('%d.0' % (index + 1))

    def add_callback(self, event, func):
        """
        Register a callback function with the list.  This function
        will be called whenever the given event occurs.

        :param event: The event that will trigger the callback
            function.  Valid events are: click1, click2, click3,
            space, return, select, up, down, next, prior, move
        :param func: The function that should be called when
            the event occurs.  ``func`` will be called with a
            single item as its argument.  (The item selected
            or the item moved to).
        """
        if event == 'select':
            events = ['click1', 'space', 'return']
        elif event == 'move':
            events = ['up', 'down', 'next', 'prior']
        else:
            events = [event]
        for e in events:
            self._callbacks.setdefault(e, {})[func] = 1

    def remove_callback(self, event, func=None):
        """
        Deregister a callback function.  If ``func`` is none, then
        all callbacks are removed for the given event.
        """
        if event is None:
            events = list(self._callbacks.keys())
        elif event == 'select':
            events = ['click1', 'space', 'return']
        elif event == 'move':
            events = ['up', 'down', 'next', 'prior']
        else:
            events = [event]
        for e in events:
            if func is None:
                del self._callbacks[e]
            else:
                try:
                    del self._callbacks[e][func]
                except:
                    pass

    def pack(self, cnf={}, **kw):
        self._itemframe.pack(cnf, **kw)

    def grid(self, cnf={}, **kw):
        self._itemframe.grid(cnf, *kw)

    def focus(self):
        self._textwidget.focus()

    def _init_itemframe(self, options):
        self._itemframe = Frame(self._parent)
        options.setdefault('background', '#e0e0e0')
        self._textwidget = Text(self._itemframe, **options)
        self._textscroll = Scrollbar(self._itemframe, takefocus=0, orient='vertical')
        self._textwidget.config(yscrollcommand=self._textscroll.set)
        self._textscroll.config(command=self._textwidget.yview)
        self._textscroll.pack(side='right', fill='y')
        self._textwidget.pack(expand=1, fill='both', side='left')
        self._textwidget.tag_config('highlight', background='#e0ffff', border='1', relief='raised')
        self._init_colortags(self._textwidget, options)
        self._textwidget.tag_config('sel', foreground='')
        self._textwidget.tag_config('sel', foreground='', background='', border='', underline=1)
        self._textwidget.tag_lower('highlight', 'sel')

    def _fire_callback(self, event, itemnum):
        if event not in self._callbacks:
            return
        if 0 <= itemnum < len(self._items):
            item = self._items[itemnum]
        else:
            item = None
        for cb_func in list(self._callbacks[event].keys()):
            cb_func(item)

    def _buttonpress(self, event):
        clickloc = '@%d,%d' % (event.x, event.y)
        insert_point = self._textwidget.index(clickloc)
        itemnum = int(insert_point.split('.')[0]) - 1
        self._fire_callback('click%d' % event.num, itemnum)

    def _keypress(self, event):
        if event.keysym == 'Return' or event.keysym == 'space':
            insert_point = self._textwidget.index('insert')
            itemnum = int(insert_point.split('.')[0]) - 1
            self._fire_callback(event.keysym.lower(), itemnum)
            return
        elif event.keysym == 'Down':
            delta = '+1line'
        elif event.keysym == 'Up':
            delta = '-1line'
        elif event.keysym == 'Next':
            delta = '+10lines'
        elif event.keysym == 'Prior':
            delta = '-10lines'
        else:
            return 'continue'
        self._textwidget.mark_set('insert', 'insert' + delta)
        self._textwidget.see('insert')
        self._textwidget.tag_remove('sel', '1.0', 'end+1char')
        self._textwidget.tag_add('sel', 'insert linestart', 'insert lineend')
        insert_point = self._textwidget.index('insert')
        itemnum = int(insert_point.split('.')[0]) - 1
        self._fire_callback(event.keysym.lower(), itemnum)
        return 'break'