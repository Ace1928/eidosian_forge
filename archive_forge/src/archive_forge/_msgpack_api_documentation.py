import gc
from . import msgpack
from .msgpack import msgpack_encoders, msgpack_decoders  # noqa: F401
from .util import force_path, FilePath, JSONInputBin, JSONOutputBin
Load a msgpack file.

    location (FilePath): The file path.
    use_list (bool): Don't use tuples instead of lists. Can make
        deserialization slower.
    RETURNS (JSONOutputBin): The loaded and deserialized content.
    