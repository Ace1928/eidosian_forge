import re
from html import escape
from html.entities import name2codepoint
from html.parser import HTMLParser
def write_pos(self):
    cur_line, cur_offset = self.getpos()
    if self.skip_output():
        self.source_pos = self.getpos()
        return
    if self.skip_next:
        self.skip_next = False
        self.source_pos = self.getpos()
        return
    if cur_line == self.source_pos[0]:
        self.write_text(self.lines[cur_line - 1][self.source_pos[1]:cur_offset])
    else:
        self.write_text(self.lines[self.source_pos[0] - 1][self.source_pos[1]:])
        self.write_text('\n')
        for i in range(self.source_pos[0] + 1, cur_line):
            self.write_text(self.lines[i - 1])
            self.write_text('\n')
        self.write_text(self.lines[cur_line - 1][:cur_offset])
    self.source_pos = self.getpos()