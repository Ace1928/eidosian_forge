from binascii import crc_hqx
from aiokeydb.v1.typing import EncodedT
Calculate key slot for a given key.
    See Keys distribution model in https://redis.io/topics/cluster-spec
    :param key - bytes
    :param bucket - int
    