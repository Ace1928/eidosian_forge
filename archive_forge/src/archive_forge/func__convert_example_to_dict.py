import struct
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Union
import numpy as np
from ray.data.block import Block
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
def _convert_example_to_dict(example: 'tf.train.Example', tf_schema: Optional['schema_pb2.Schema']) -> Dict[str, 'pyarrow.Array']:
    record = {}
    schema_dict = {}
    if tf_schema is not None:
        for schema_feature in tf_schema.feature:
            schema_dict[schema_feature.name] = schema_feature.type
    for feature_name, feature in example.features.feature.items():
        if tf_schema is not None and feature_name not in schema_dict:
            raise ValueError(f'Found extra unexpected feature {feature_name} not in specified schema: {tf_schema}')
        schema_feature_type = schema_dict.get(feature_name)
        record[feature_name] = _get_feature_value(feature, schema_feature_type)
    return record