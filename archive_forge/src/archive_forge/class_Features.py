import copy
import json
import re
import sys
from collections.abc import Iterable, Mapping
from collections.abc import Sequence as SequenceABC
from dataclasses import InitVar, dataclass, field, fields
from functools import reduce, wraps
from operator import mul
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
from typing import Sequence as Sequence_
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
import pyarrow_hotfix  # noqa: F401  # to fix vulnerability on pyarrow<14.0.1
from pandas.api.extensions import ExtensionArray as PandasExtensionArray
from pandas.api.extensions import ExtensionDtype as PandasExtensionDtype
from .. import config
from ..naming import camelcase_to_snakecase, snakecase_to_camelcase
from ..table import array_cast
from ..utils import logging
from ..utils.py_utils import asdict, first_non_null_value, zip_dict
from .audio import Audio
from .image import Image, encode_pil_image
from .translation import Translation, TranslationVariableLanguages
class Features(dict):
    """A special dictionary that defines the internal structure of a dataset.

    Instantiated with a dictionary of type `dict[str, FieldType]`, where keys are the desired column names,
    and values are the type of that column.

    `FieldType` can be one of the following:
        - a [`~datasets.Value`] feature specifies a single typed value, e.g. `int64` or `string`.
        - a [`~datasets.ClassLabel`] feature specifies a field with a predefined set of classes which can have labels
          associated to them and will be stored as integers in the dataset.
        - a python `dict` which specifies that the field is a nested field containing a mapping of sub-fields to sub-fields
          features. It's possible to have nested fields of nested fields in an arbitrary manner.
        - a python `list` or a [`~datasets.Sequence`] specifies that the field contains a list of objects. The python
          `list` or [`~datasets.Sequence`] should be provided with a single sub-feature as an example of the feature
          type hosted in this list.

          <Tip>

           A [`~datasets.Sequence`] with a internal dictionary feature will be automatically converted into a dictionary of
           lists. This behavior is implemented to have a compatilbity layer with the TensorFlow Datasets library but may be
           un-wanted in some cases. If you don't want this behavior, you can use a python `list` instead of the
           [`~datasets.Sequence`].

          </Tip>

        - a [`Array2D`], [`Array3D`], [`Array4D`] or [`Array5D`] feature for multidimensional arrays.
        - an [`Audio`] feature to store the absolute path to an audio file or a dictionary with the relative path
          to an audio file ("path" key) and its bytes content ("bytes" key). This feature extracts the audio data.
        - an [`Image`] feature to store the absolute path to an image file, an `np.ndarray` object, a `PIL.Image.Image` object
          or a dictionary with the relative path to an image file ("path" key) and its bytes content ("bytes" key). This feature extracts the image data.
        - [`~datasets.Translation`] and [`~datasets.TranslationVariableLanguages`], the two features specific to Machine Translation.
    """

    def __init__(*args, **kwargs):
        if not args:
            raise TypeError("descriptor '__init__' of 'Features' object needs an argument")
        self, *args = args
        super(Features, self).__init__(*args, **kwargs)
        self._column_requires_decoding: Dict[str, bool] = {col: require_decoding(feature) for col, feature in self.items()}
    __setitem__ = keep_features_dicts_synced(dict.__setitem__)
    __delitem__ = keep_features_dicts_synced(dict.__delitem__)
    update = keep_features_dicts_synced(dict.update)
    setdefault = keep_features_dicts_synced(dict.setdefault)
    pop = keep_features_dicts_synced(dict.pop)
    popitem = keep_features_dicts_synced(dict.popitem)
    clear = keep_features_dicts_synced(dict.clear)

    def __reduce__(self):
        return (Features, (dict(self),))

    @property
    def type(self):
        """
        Features field types.

        Returns:
            :obj:`pyarrow.DataType`
        """
        return get_nested_type(self)

    @property
    def arrow_schema(self):
        """
        Features schema.

        Returns:
            :obj:`pyarrow.Schema`
        """
        hf_metadata = {'info': {'features': self.to_dict()}}
        return pa.schema(self.type).with_metadata({'huggingface': json.dumps(hf_metadata)})

    @classmethod
    def from_arrow_schema(cls, pa_schema: pa.Schema) -> 'Features':
        """
        Construct [`Features`] from Arrow Schema.
        It also checks the schema metadata for Hugging Face Datasets features.
        Non-nullable fields are not supported and set to nullable.

        Args:
            pa_schema (`pyarrow.Schema`):
                Arrow Schema.

        Returns:
            [`Features`]
        """
        metadata_features = Features()
        if pa_schema.metadata is not None and 'huggingface'.encode('utf-8') in pa_schema.metadata:
            metadata = json.loads(pa_schema.metadata['huggingface'.encode('utf-8')].decode())
            if 'info' in metadata and 'features' in metadata['info'] and (metadata['info']['features'] is not None):
                metadata_features = Features.from_dict(metadata['info']['features'])
        metadata_features_schema = metadata_features.arrow_schema
        obj = {field.name: metadata_features[field.name] if field.name in metadata_features and metadata_features_schema.field(field.name) == field else generate_from_arrow_type(field.type) for field in pa_schema}
        return cls(**obj)

    @classmethod
    def from_dict(cls, dic) -> 'Features':
        """
        Construct [`Features`] from dict.

        Regenerate the nested feature object from a deserialized dict.
        We use the `_type` key to infer the dataclass name of the feature `FieldType`.

        It allows for a convenient constructor syntax
        to define features from deserialized JSON dictionaries. This function is used in particular when deserializing
        a [`DatasetInfo`] that was dumped to a JSON object. This acts as an analogue to
        [`Features.from_arrow_schema`] and handles the recursive field-by-field instantiation, but doesn't require
        any mapping to/from pyarrow, except for the fact that it takes advantage of the mapping of pyarrow primitive
        dtypes that [`Value`] automatically performs.

        Args:
            dic (`dict[str, Any]`):
                Python dictionary.

        Returns:
            `Features`

        Example::
            >>> Features.from_dict({'_type': {'dtype': 'string', 'id': None, '_type': 'Value'}})
            {'_type': Value(dtype='string', id=None)}
        """
        obj = generate_from_dict(dic)
        return cls(**obj)

    def to_dict(self):
        return asdict(self)

    def _to_yaml_list(self) -> list:
        yaml_data = self.to_dict()

        def simplify(feature: dict) -> dict:
            if not isinstance(feature, dict):
                raise TypeError(f'Expected a dict but got a {type(feature)}: {feature}')
            if isinstance(feature.get('sequence'), dict) and list(feature['sequence']) == ['dtype']:
                feature['sequence'] = feature['sequence']['dtype']
            if isinstance(feature.get('sequence'), dict) and list(feature['sequence']) == ['struct']:
                feature['sequence'] = feature['sequence']['struct']
            if isinstance(feature.get('list'), dict) and list(feature['list']) == ['dtype']:
                feature['list'] = feature['list']['dtype']
            if isinstance(feature.get('list'), dict) and list(feature['list']) == ['struct']:
                feature['list'] = feature['list']['struct']
            if isinstance(feature.get('class_label'), dict) and isinstance(feature['class_label'].get('names'), list):
                feature['class_label']['names'] = {str(label_id): label_name for label_id, label_name in enumerate(feature['class_label']['names'])}
            return feature

        def to_yaml_inner(obj: Union[dict, list]) -> dict:
            if isinstance(obj, dict):
                _type = obj.pop('_type', None)
                if _type == 'Sequence':
                    _feature = obj.pop('feature')
                    return simplify({'sequence': to_yaml_inner(_feature), **obj})
                elif _type == 'Value':
                    return obj
                elif _type and (not obj):
                    return {'dtype': camelcase_to_snakecase(_type)}
                elif _type:
                    return {'dtype': simplify({camelcase_to_snakecase(_type): obj})}
                else:
                    return {'struct': [{'name': name, **to_yaml_inner(_feature)} for name, _feature in obj.items()]}
            elif isinstance(obj, list):
                return simplify({'list': simplify(to_yaml_inner(obj[0]))})
            elif isinstance(obj, tuple):
                return to_yaml_inner(list(obj))
            else:
                raise TypeError(f'Expected a dict or a list but got {type(obj)}: {obj}')

        def to_yaml_types(obj: dict) -> dict:
            if isinstance(obj, dict):
                return {k: to_yaml_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_yaml_types(v) for v in obj]
            elif isinstance(obj, tuple):
                return to_yaml_types(list(obj))
            else:
                return obj
        return to_yaml_types(to_yaml_inner(yaml_data)['struct'])

    @classmethod
    def _from_yaml_list(cls, yaml_data: list) -> 'Features':
        yaml_data = copy.deepcopy(yaml_data)

        def unsimplify(feature: dict) -> dict:
            if not isinstance(feature, dict):
                raise TypeError(f'Expected a dict but got a {type(feature)}: {feature}')
            if isinstance(feature.get('sequence'), str):
                feature['sequence'] = {'dtype': feature['sequence']}
            if isinstance(feature.get('list'), str):
                feature['list'] = {'dtype': feature['list']}
            if isinstance(feature.get('class_label'), dict) and isinstance(feature['class_label'].get('names'), dict):
                label_ids = sorted(feature['class_label']['names'], key=int)
                if label_ids and [int(label_id) for label_id in label_ids] != list(range(int(label_ids[-1]) + 1)):
                    raise ValueError(f'ClassLabel expected a value for all label ids [0:{int(label_ids[-1]) + 1}] but some ids are missing.')
                feature['class_label']['names'] = [feature['class_label']['names'][label_id] for label_id in label_ids]
            return feature

        def from_yaml_inner(obj: Union[dict, list]) -> Union[dict, list]:
            if isinstance(obj, dict):
                if not obj:
                    return {}
                _type = next(iter(obj))
                if _type == 'sequence':
                    _feature = unsimplify(obj).pop(_type)
                    return {'feature': from_yaml_inner(_feature), **obj, '_type': 'Sequence'}
                if _type == 'list':
                    return [from_yaml_inner(unsimplify(obj)[_type])]
                if _type == 'struct':
                    return from_yaml_inner(obj['struct'])
                elif _type == 'dtype':
                    if isinstance(obj['dtype'], str):
                        try:
                            Value(obj['dtype'])
                            return {**obj, '_type': 'Value'}
                        except ValueError:
                            return {'_type': snakecase_to_camelcase(obj['dtype'])}
                    else:
                        return from_yaml_inner(obj['dtype'])
                else:
                    return {'_type': snakecase_to_camelcase(_type), **unsimplify(obj)[_type]}
            elif isinstance(obj, list):
                names = [_feature.pop('name') for _feature in obj]
                return {name: from_yaml_inner(_feature) for name, _feature in zip(names, obj)}
            else:
                raise TypeError(f'Expected a dict or a list but got {type(obj)}: {obj}')
        return cls.from_dict(from_yaml_inner(yaml_data))

    def encode_example(self, example):
        """
        Encode example into a format for Arrow.

        Args:
            example (`dict[str, Any]`):
                Data in a Dataset row.

        Returns:
            `dict[str, Any]`
        """
        example = cast_to_python_objects(example)
        return encode_nested_example(self, example)

    def encode_column(self, column, column_name: str):
        """
        Encode column into a format for Arrow.

        Args:
            column (`list[Any]`):
                Data in a Dataset column.
            column_name (`str`):
                Dataset column name.

        Returns:
            `list[Any]`
        """
        column = cast_to_python_objects(column)
        return [encode_nested_example(self[column_name], obj) for obj in column]

    def encode_batch(self, batch):
        """
        Encode batch into a format for Arrow.

        Args:
            batch (`dict[str, list[Any]]`):
                Data in a Dataset batch.

        Returns:
            `dict[str, list[Any]]`
        """
        encoded_batch = {}
        if set(batch) != set(self):
            raise ValueError(f'Column mismatch between batch {set(batch)} and features {set(self)}')
        for key, column in batch.items():
            column = cast_to_python_objects(column)
            encoded_batch[key] = [encode_nested_example(self[key], obj) for obj in column]
        return encoded_batch

    def decode_example(self, example: dict, token_per_repo_id: Optional[Dict[str, Union[str, bool, None]]]=None):
        """Decode example with custom feature decoding.

        Args:
            example (`dict[str, Any]`):
                Dataset row data.
            token_per_repo_id (`dict`, *optional*):
                To access and decode audio or image files from private repositories on the Hub, you can pass
                a dictionary `repo_id (str) -> token (bool or str)`.

        Returns:
            `dict[str, Any]`
        """
        return {column_name: decode_nested_example(feature, value, token_per_repo_id=token_per_repo_id) if self._column_requires_decoding[column_name] else value for column_name, (feature, value) in zip_dict({key: value for key, value in self.items() if key in example}, example)}

    def decode_column(self, column: list, column_name: str):
        """Decode column with custom feature decoding.

        Args:
            column (`list[Any]`):
                Dataset column data.
            column_name (`str`):
                Dataset column name.

        Returns:
            `list[Any]`
        """
        return [decode_nested_example(self[column_name], value) if value is not None else None for value in column] if self._column_requires_decoding[column_name] else column

    def decode_batch(self, batch: dict, token_per_repo_id: Optional[Dict[str, Union[str, bool, None]]]=None):
        """Decode batch with custom feature decoding.

        Args:
            batch (`dict[str, list[Any]]`):
                Dataset batch data.
            token_per_repo_id (`dict`, *optional*):
                To access and decode audio or image files from private repositories on the Hub, you can pass
                a dictionary repo_id (str) -> token (bool or str)

        Returns:
            `dict[str, list[Any]]`
        """
        decoded_batch = {}
        for column_name, column in batch.items():
            decoded_batch[column_name] = [decode_nested_example(self[column_name], value, token_per_repo_id=token_per_repo_id) if value is not None else None for value in column] if self._column_requires_decoding[column_name] else column
        return decoded_batch

    def copy(self) -> 'Features':
        """
        Make a deep copy of [`Features`].

        Returns:
            [`Features`]

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="train")
        >>> copy_of_features = ds.features.copy()
        >>> copy_of_features
        {'label': ClassLabel(num_classes=2, names=['neg', 'pos'], id=None),
         'text': Value(dtype='string', id=None)}
        ```
        """
        return copy.deepcopy(self)

    def reorder_fields_as(self, other: 'Features') -> 'Features':
        """
        Reorder Features fields to match the field order of other [`Features`].

        The order of the fields is important since it matters for the underlying arrow data.
        Re-ordering the fields allows to make the underlying arrow data type match.

        Args:
            other ([`Features`]):
                The other [`Features`] to align with.

        Returns:
            [`Features`]

        Example::

            >>> from datasets import Features, Sequence, Value
            >>> # let's say we have to features with a different order of nested fields (for a and b for example)
            >>> f1 = Features({"root": Sequence({"a": Value("string"), "b": Value("string")})})
            >>> f2 = Features({"root": {"b": Sequence(Value("string")), "a": Sequence(Value("string"))}})
            >>> assert f1.type != f2.type
            >>> # re-ordering keeps the base structure (here Sequence is defined at the root level), but make the fields order match
            >>> f1.reorder_fields_as(f2)
            {'root': Sequence(feature={'b': Value(dtype='string', id=None), 'a': Value(dtype='string', id=None)}, length=-1, id=None)}
            >>> assert f1.reorder_fields_as(f2).type == f2.type
        """

        def recursive_reorder(source, target, stack=''):
            stack_position = ' at ' + stack[1:] if stack else ''
            if isinstance(target, Sequence):
                target = target.feature
                if isinstance(target, dict):
                    target = {k: [v] for k, v in target.items()}
                else:
                    target = [target]
            if isinstance(source, Sequence):
                source, id_, length = (source.feature, source.id, source.length)
                if isinstance(source, dict):
                    source = {k: [v] for k, v in source.items()}
                    reordered = recursive_reorder(source, target, stack)
                    return Sequence({k: v[0] for k, v in reordered.items()}, id=id_, length=length)
                else:
                    source = [source]
                    reordered = recursive_reorder(source, target, stack)
                    return Sequence(reordered[0], id=id_, length=length)
            elif isinstance(source, dict):
                if not isinstance(target, dict):
                    raise ValueError(f'Type mismatch: between {source} and {target}' + stack_position)
                if sorted(source) != sorted(target):
                    message = f'Keys mismatch: between {source} (source) and {target} (target).\n{source.keys() - target.keys()} are missing from target and {target.keys() - source.keys()} are missing from source' + stack_position
                    raise ValueError(message)
                return {key: recursive_reorder(source[key], target[key], stack + f'.{key}') for key in target}
            elif isinstance(source, list):
                if not isinstance(target, list):
                    raise ValueError(f'Type mismatch: between {source} and {target}' + stack_position)
                if len(source) != len(target):
                    raise ValueError(f'Length mismatch: between {source} and {target}' + stack_position)
                return [recursive_reorder(source[i], target[i], stack + '.<list>') for i in range(len(target))]
            else:
                return source
        return Features(recursive_reorder(self, other))

    def flatten(self, max_depth=16) -> 'Features':
        """Flatten the features. Every dictionary column is removed and is replaced by
        all the subfields it contains. The new fields are named by concatenating the
        name of the original column and the subfield name like this: `<original>.<subfield>`.

        If a column contains nested dictionaries, then all the lower-level subfields names are
        also concatenated to form new columns: `<original>.<subfield>.<subsubfield>`, etc.

        Returns:
            [`Features`]:
                The flattened features.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("squad", split="train")
        >>> ds.features.flatten()
        {'answers.answer_start': Sequence(feature=Value(dtype='int32', id=None), length=-1, id=None),
         'answers.text': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
         'context': Value(dtype='string', id=None),
         'id': Value(dtype='string', id=None),
         'question': Value(dtype='string', id=None),
         'title': Value(dtype='string', id=None)}
        ```
        """
        for depth in range(1, max_depth):
            no_change = True
            flattened = self.copy()
            for column_name, subfeature in self.items():
                if isinstance(subfeature, dict):
                    no_change = False
                    flattened.update({f'{column_name}.{k}': v for k, v in subfeature.items()})
                    del flattened[column_name]
                elif isinstance(subfeature, Sequence) and isinstance(subfeature.feature, dict):
                    no_change = False
                    flattened.update({f'{column_name}.{k}': Sequence(v) if not isinstance(v, dict) else [v] for k, v in subfeature.feature.items()})
                    del flattened[column_name]
                elif hasattr(subfeature, 'flatten') and subfeature.flatten() != subfeature:
                    no_change = False
                    flattened.update({f'{column_name}.{k}': v for k, v in subfeature.flatten().items()})
                    del flattened[column_name]
            self = flattened
            if no_change:
                break
        return self