import abc
import logging
import os
import random
import shutil
import time
import urllib
import uuid
from collections import namedtuple
from typing import IO, List, Optional, Tuple
import ray
from ray._private.ray_constants import DEFAULT_OBJECT_PREFIX
from ray._raylet import ObjectRef
def _size_check(self, address_len, metadata_len, buffer_len, obtained_data_size):
    """Check whether or not the obtained_data_size is as expected.

        Args:
             metadata_len: Actual metadata length of the object.
             buffer_len: Actual buffer length of the object.
             obtained_data_size: Data size specified in the
                url_with_offset.

        Raises:
            ValueError if obtained_data_size is different from
            address_len + metadata_len + buffer_len +
            24 (first 8 bytes to store length).
        """
    data_size_in_bytes = address_len + metadata_len + buffer_len + self.HEADER_LENGTH
    if data_size_in_bytes != obtained_data_size:
        raise ValueError(f'Obtained data has a size of {data_size_in_bytes}, although it is supposed to have the size of {obtained_data_size}.')