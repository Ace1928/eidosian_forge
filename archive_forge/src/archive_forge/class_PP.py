import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
class PP:

    def __init__(self):
        self.max_lines = 200
        self.max_width = 60
        self.bounded = False
        self.max_indent = 40

    def pp_string(self, f, indent):
        if not self.bounded or self.pos <= self.max_width:
            sz = _len(f)
            if self.bounded and self.pos + sz > self.max_width:
                self.out.write(u(_ellipses))
            else:
                self.pos = self.pos + sz
                self.ribbon_pos = self.ribbon_pos + sz
                self.out.write(u(f.string))

    def pp_compose(self, f, indent):
        for c in f.children:
            self.pp(c, indent)

    def pp_choice(self, f, indent):
        space_left = self.max_width - self.pos
        if space_left > 0 and fits(f.children[0], space_left):
            self.pp(f.children[0], indent)
        else:
            self.pp(f.children[1], indent)

    def pp_line_break(self, f, indent):
        self.pos = indent
        self.ribbon_pos = 0
        self.line = self.line + 1
        if self.line < self.max_lines:
            self.out.write(u('\n'))
            for i in range(indent):
                self.out.write(u(' '))
        else:
            self.out.write(u('\n...'))
            raise StopPPException()

    def pp(self, f, indent):
        if isinstance(f, str):
            self.pp_string(f, indent)
        elif f.is_string():
            self.pp_string(f, indent)
        elif f.is_indent():
            self.pp(f.child, min(indent + f.indent, self.max_indent))
        elif f.is_compose():
            self.pp_compose(f, indent)
        elif f.is_choice():
            self.pp_choice(f, indent)
        elif f.is_linebreak():
            self.pp_line_break(f, indent)
        else:
            return

    def __call__(self, out, f):
        try:
            self.pos = 0
            self.ribbon_pos = 0
            self.line = 0
            self.out = out
            self.pp(f, 0)
        except StopPPException:
            return