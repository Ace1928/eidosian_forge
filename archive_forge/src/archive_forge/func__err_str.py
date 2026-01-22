import unicodedata
def _err_str(self):
    lineno, colno = self._err_offsets()
    if self.errpos == len(self.msg):
        thing = 'end of input'
    else:
        thing = f'"{self.msg[self.errpos]}"'
    return f'{self.fname}:{lineno} Unexpected {thing} at column {colno}'