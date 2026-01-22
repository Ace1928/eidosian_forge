from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
class ShowText:
    """
    A ``Tkinter`` window used to display a text.  ``ShowText`` is
    typically used by graphical tools to display help text, or similar
    information.
    """

    def __init__(self, root, title, text, width=None, height=None, **textbox_options):
        if width is None or height is None:
            width, height = self.find_dimentions(text, width, height)
        if root is None:
            self._top = top = Tk()
        else:
            self._top = top = Toplevel(root)
        top.title(title)
        b = Button(top, text='Ok', command=self.destroy)
        b.pack(side='bottom')
        tbf = Frame(top)
        tbf.pack(expand=1, fill='both')
        scrollbar = Scrollbar(tbf, orient='vertical')
        scrollbar.pack(side='right', fill='y')
        textbox = Text(tbf, wrap='word', width=width, height=height, **textbox_options)
        textbox.insert('end', text)
        textbox['state'] = 'disabled'
        textbox.pack(side='left', expand=1, fill='both')
        scrollbar['command'] = textbox.yview
        textbox['yscrollcommand'] = scrollbar.set
        top.bind('q', self.destroy)
        top.bind('x', self.destroy)
        top.bind('c', self.destroy)
        top.bind('<Return>', self.destroy)
        top.bind('<Escape>', self.destroy)
        scrollbar.focus()

    def find_dimentions(self, text, width, height):
        lines = text.split('\n')
        if width is None:
            maxwidth = max((len(line) for line in lines))
            width = min(maxwidth, 80)
        height = 0
        for line in lines:
            while len(line) > width:
                brk = line[:width].rfind(' ')
                line = line[brk:]
                height += 1
            height += 1
        height = min(height, 25)
        return (width, height)

    def destroy(self, *e):
        if self._top is None:
            return
        self._top.destroy()
        self._top = None

    def mainloop(self, *args, **kwargs):
        """
        Enter the Tkinter mainloop.  This function must be called if
        this window is created from a non-interactive program (e.g.
        from a secript); otherwise, the window will close as soon as
        the script completes.
        """
        if in_idle():
            return
        self._top.mainloop(*args, **kwargs)