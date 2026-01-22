import uuid
import numpy as np
from ..data_utils import is_pandas_df, has_geo_interface, records_from_geo_interface
from .json_tools import JSONMixin, camel_and_lower
from ..settings import settings as pydeck_settings
from pydeck.types import Image, Function
from pydeck.exceptions import BinaryTransportException
def _prepare_binary_data(self, data_set):
    if not is_pandas_df(data_set):
        raise BinaryTransportException('Layer data must be a `pandas.DataFrame` type')
    layer_accessors = self._kwargs
    inverted_accessor_map = {v: k for k, v in layer_accessors.items() if type(v) not in [list, dict, set]}
    binary_transmission = []
    for column in data_set.columns:
        np_data = np.stack(data_set[column].to_numpy())
        del self.__dict__[inverted_accessor_map[column]]
        binary_transmission.append({'layer_id': self.id, 'column_name': column, 'accessor': camel_and_lower(inverted_accessor_map[column]), 'np_data': np_data})
    return binary_transmission