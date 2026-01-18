import hashlib
import zlib
from .. import estimate_compressed_size, tests
We generate some hex-data that can be seeded.

        The output should be deterministic, but the data stream is effectively
        random.
        