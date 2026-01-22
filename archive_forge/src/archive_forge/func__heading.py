import logging
def _heading(self, s, border_char):
    border = border_char * len(s)
    self.new_paragraph()
    self.doc.write(f'{border}\n{s}\n{border}')
    self.new_paragraph()