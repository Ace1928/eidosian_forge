import struct
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c
Open a file to keep the tensorboard records.

        Args:
        writer: A file-like object that implements `write`, `flush` and `close`.
        