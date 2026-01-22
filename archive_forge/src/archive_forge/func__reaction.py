from Bio.KEGG import _default_wrap, _struct_wrap, _wrap_kegg, _write_kegg
def _reaction(self):
    return _write_kegg('REACTION', [_wrap_kegg(line, wrap_rule=rxn_wrap) for line in self.reaction])