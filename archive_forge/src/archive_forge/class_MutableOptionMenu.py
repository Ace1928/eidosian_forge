from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
class MutableOptionMenu(Menubutton):

    def __init__(self, master, values, **options):
        self._callback = options.get('command')
        if 'command' in options:
            del options['command']
        self._variable = variable = StringVar()
        if len(values) > 0:
            variable.set(values[0])
        kw = {'borderwidth': 2, 'textvariable': variable, 'indicatoron': 1, 'relief': RAISED, 'anchor': 'c', 'highlightthickness': 2}
        kw.update(options)
        Widget.__init__(self, master, 'menubutton', kw)
        self.widgetName = 'tk_optionMenu'
        self._menu = Menu(self, name='menu', tearoff=0)
        self.menuname = self._menu._w
        self._values = []
        for value in values:
            self.add(value)
        self['menu'] = self._menu

    def add(self, value):
        if value in self._values:
            return

        def set(value=value):
            self.set(value)
        self._menu.add_command(label=value, command=set)
        self._values.append(value)

    def set(self, value):
        self._variable.set(value)
        if self._callback:
            self._callback(value)

    def remove(self, value):
        i = self._values.index(value)
        del self._values[i]
        self._menu.delete(i, i)

    def __getitem__(self, name):
        if name == 'menu':
            return self.__menu
        return Widget.__getitem__(self, name)

    def destroy(self):
        """Destroy this widget and the associated menu."""
        Menubutton.destroy(self)
        self._menu = None