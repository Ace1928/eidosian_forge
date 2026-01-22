from Bio.KEGG import _default_wrap, _struct_wrap, _wrap_kegg, _write_kegg
def _classname(self):
    return _write_kegg('CLASS', self.classname)