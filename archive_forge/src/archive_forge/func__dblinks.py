from Bio.KEGG import _default_wrap, _wrap_kegg, _write_kegg
def _dblinks(self):
    s = []
    for entry in self.dblinks:
        s.append(entry[0] + ': ' + ' '.join(entry[1]))
    return _write_kegg('DBLINKS', [_wrap_kegg(line, wrap_rule=id_wrap(9)) for line in s])