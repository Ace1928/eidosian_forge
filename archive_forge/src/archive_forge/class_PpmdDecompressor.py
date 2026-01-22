from typing import Union
class PpmdDecompressor:
    """Decompressor class to decompress data by PPMd algorithm."""

    def __init__(self, max_order: int=6, mem_size: int=8 << 20, *, restore_method=PPMD8_RESTORE_METHOD_RESTART, variant: str='I'):
        if variant not in ['H', 'I', 'h', 'i']:
            raise ValueError('Unsupported PPMd variant')
        if variant in ['I', 'i']:
            self.decoder = Ppmd8Decoder(max_order=max_order, mem_size=mem_size, restore_method=restore_method)
        else:
            self.decoder = Ppmd7Decoder(max_order=max_order, mem_size=mem_size)
        self.eof = False
        self.need_input = True

    def decompress(self, data: Union[bytes, memoryview]):
        if self.decoder.eof:
            self.eof = True
            return b''
        if self.decoder.needs_input and len(data) == 0:
            raise PpmdError('No enough data is provided for decompression.')
        elif not self.decoder.needs_input and len(data) > 0:
            raise PpmdError('Unused data is given.')
        return self.decoder.decode(data)