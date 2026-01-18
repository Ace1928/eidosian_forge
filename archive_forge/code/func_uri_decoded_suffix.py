from __future__ import annotations
@property
def uri_decoded_suffix(self) -> Optional[str]:
    try:
        return self._uri_decoded_suffix
    except AttributeError:
        pass
    if self.suffix is None:
        self._uri_decoded_suffix: Optional[str] = None
        return None
    res = ''
    idx = 0
    while idx < len(self.suffix):
        ch = self.suffix[idx]
        idx += 1
        if ch != '%':
            res += ch
        else:
            res += chr(int(self.suffix[idx:idx + 2], 16))
            idx += 2
    self._uri_decoded_suffix = res
    return res