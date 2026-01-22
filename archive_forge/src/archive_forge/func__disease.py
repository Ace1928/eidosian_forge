from Bio.KEGG import _default_wrap, _struct_wrap, _wrap_kegg, _write_kegg
def _disease(self):
    s = []
    for entry in self.disease:
        s.append(entry[0] + ': ' + entry[1] + '  ' + entry[2])
    return _write_kegg('DISEASE', [_wrap_kegg(line, wrap_rule=id_wrap(13)) for line in s])