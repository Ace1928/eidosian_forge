import warnings
def _AddSymbol(self, name, file_desc_proto):
    if name in self._file_desc_protos_by_symbol:
        warn_msg = 'Conflict register for file "' + file_desc_proto.name + '": ' + name + ' is already defined in file "' + self._file_desc_protos_by_symbol[name].name + '"'
        warnings.warn(warn_msg, RuntimeWarning)
    self._file_desc_protos_by_symbol[name] = file_desc_proto