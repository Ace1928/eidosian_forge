from typing import Union
from huggingface_hub.utils import insecure_hashlib
class KeyHasher:
    """KeyHasher class for providing hash using md5"""

    def __init__(self, hash_salt: str):
        self._split_md5 = insecure_hashlib.md5(_as_bytes(hash_salt))

    def hash(self, key: Union[str, int, bytes]) -> int:
        """Returns 128-bits unique hash of input key

        Args:
        key: the input key to be hashed (should be str, int or bytes)

        Returns: 128-bit int hash key"""
        md5 = self._split_md5.copy()
        byte_key = _as_bytes(key)
        md5.update(byte_key)
        return int(md5.hexdigest(), 16)