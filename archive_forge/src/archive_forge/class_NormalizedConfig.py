import functools
from typing import Callable, Dict, Type, Union
from transformers import PretrainedConfig
class NormalizedConfig:
    """
    Handles the normalization of [`PretrainedConfig`] attribute names, allowing to access attributes in a general way.

    Attributes:
        config ([`PretrainedConfig`]):
            The config to normalize.
    """

    def __init__(self, config: Union[PretrainedConfig, Dict], allow_new: bool=False, **kwargs):
        self.config = config
        for key, value in kwargs.items():
            if allow_new or hasattr(self, key.upper()):
                setattr(self, key.upper(), value)
            else:
                raise AttributeError(f'{self.__class__} has not attribute {key}. Set allow_new=True to add a new attribute.')

    @classmethod
    def with_args(cls, allow_new: bool=False, **kwargs) -> Callable[[PretrainedConfig], 'NormalizedConfig']:
        return functools.partial(cls, allow_new=allow_new, **kwargs)

    def __getattr__(self, attr_name):
        if attr_name == 'config':
            return super().__getattr__(attr_name)
        try:
            attr_name = super().__getattribute__(attr_name.upper())
        except AttributeError:
            pass
        attr_name = attr_name.split('.')
        leaf_attr_name = attr_name[-1]
        config = self.config
        for attr in attr_name[:-1]:
            config = getattr(config, attr)
        attr = getattr(config, leaf_attr_name, None)
        if attr is None:
            attribute_map = getattr(self.config, 'attribute_map', {})
            attr = getattr(self.config, attribute_map.get(leaf_attr_name, ''), None)
        if attr is None:
            raise AttributeError(f'Could not find the attribute named "{leaf_attr_name}" in the normalized config.')
        return attr

    def has_attribute(self, attr_name):
        try:
            self.__getattr__(attr_name)
        except AttributeError:
            return False
        return True