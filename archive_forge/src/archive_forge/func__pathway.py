from Bio.KEGG import _default_wrap, _struct_wrap, _wrap_kegg, _write_kegg
def _pathway(self):
    s = []
    for entry in self.pathway:
        s.append(entry[0] + '  ' + entry[1])
    return _write_kegg('PATHWAY', [_wrap_kegg(line, wrap_rule=id_wrap(16)) for line in s])