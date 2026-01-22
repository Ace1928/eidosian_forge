from .gui import Tk_, ttk, SimpleDialog
class InfoDialog(baseclass):

    def __init__(self, parent, title, style, content=''):
        self.parent, self.style, self.content = (parent, style, content)
        Tk_.Toplevel.__init__(self, parent)
        if title:
            self.title(title)
        text = Tk_.Text(self, font=style.font, width=50, height=18, padx=10, relief=Tk_.FLAT, background=style.windowBG, highlightthickness=0)
        text.insert(Tk_.END, self.content)
        text.focus_set()
        text.config(state=Tk_.DISABLED)
        text.grid(row=0, column=1, sticky=Tk_.N + Tk_.W, padx=10, pady=10)
        self.buttonbox()
        self.grab_set()
        self.protocol('WM_DELETE_WINDOW', self.ok)
        self.focus_set()
        self.wait_window(self)

    def buttonbox(self):
        box = ttk.Frame(self)
        w = ttk.Button(box, text='OK', width=10, command=self.ok, default=Tk_.ACTIVE)
        w.pack(side=Tk_.LEFT, padx=5, pady=5)
        self.bind('<Return>', self.ok)
        self.bind('<Escape>', self.ok)
        box.grid(row=1, columnspan=2)

    def ok(self, event=None):
        self.parent.focus_set()
        self.app = None
        self.destroy()