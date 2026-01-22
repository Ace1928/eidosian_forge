from typing import Union
class PpmdCompressor:
    """Compressor class to compress data by PPMd algorithm."""

    def __init__(self, max_order: int=6, mem_size: int=8 << 20, *, restore_method=PPMD8_RESTORE_METHOD_RESTART, variant: str='I'):
        if variant not in ['H', 'I', 'h', 'i']:
            raise ValueError('Unsupported PPMd variant')
        if variant in ['I', 'i']:
            self.encoder = Ppmd8Encoder(max_order, mem_size, restore_method)
        else:
            self.encoder = Ppmd7Encoder(max_order, mem_size)
        self.eof = False

    def compress(self, data_or_str: Union[bytes, bytearray, memoryview, str]):
        if type(data_or_str) == str:
            data = data_or_str.encode('UTF-8')
        elif _is_bytelike(data_or_str):
            data = data_or_str
        else:
            raise ValueError('Argument data_or_str is neither bytes-like object nor str.')
        return self.encoder.encode(data)

    def flush(self):
        self.eof = True
        return self.encoder.flush()