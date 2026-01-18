import logging
def tocitem(self, item, file_name=None):
    if self.doc.target == 'man':
        self.li(item)
    elif file_name:
        self.doc.writeln('  %s' % file_name)
    else:
        self.doc.writeln('  %s' % item)