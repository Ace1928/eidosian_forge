import logging
def write_py_doc_string(self, docstring):
    docstring_lines = docstring.splitlines()
    for docstring_line in docstring_lines:
        self.doc.writeln(docstring_line)