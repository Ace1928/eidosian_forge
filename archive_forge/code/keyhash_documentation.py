from typing import Union
from huggingface_hub.utils import insecure_hashlib
Returns 128-bits unique hash of input key

        Args:
        key: the input key to be hashed (should be str, int or bytes)

        Returns: 128-bit int hash key