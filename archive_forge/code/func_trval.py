from __future__ import annotations
@property
def trval(self) -> Optional[str]:
    try:
        return self._trval
    except AttributeError:
        pass
    if self.handle is None:
        self._trval: Optional[str] = self.uri_decoded_suffix
        return self._trval
    assert self._transform_type is not None
    if not self._transform_type:
        self._trval = self.handles[self.handle] + self.uri_decoded_suffix
        return self._trval
    if self.handle == '!!' and self.suffix in ('null', 'bool', 'int', 'float', 'binary', 'timestamp', 'omap', 'pairs', 'set', 'str', 'seq', 'map'):
        self._trval = self.handles[self.handle] + self.uri_decoded_suffix
    else:
        self._trval = self.handles[self.handle] + self.uri_decoded_suffix
    return self._trval