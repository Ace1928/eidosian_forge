from Bio.KEGG import _default_wrap, _struct_wrap, _wrap_kegg, _write_kegg
def _formula(self):
    return _write_kegg('FORMULA', [self.formula])