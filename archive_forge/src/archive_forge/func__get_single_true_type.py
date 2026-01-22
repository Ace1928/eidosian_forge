import struct
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Union
import numpy as np
from ray.data.block import Block
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
def _get_single_true_type(dct) -> str:
    """Utility function for getting the single key which has a `True` value in
    a dict. Used to filter a dict of `{field_type: is_valid}` to get
    the field type from a schema or data source."""
    filtered_types = iter([_type for _type in dct if dct[_type]])
    return next(filtered_types, None)