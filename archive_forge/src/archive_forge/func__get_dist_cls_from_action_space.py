import enum
import functools
from typing import Optional
import gymnasium as gym
import numpy as np
import tree
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
from ray.rllib.core.models.base import Encoder
from ray.rllib.core.models.configs import (
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.models.distributions import Distribution
from ray.rllib.models.preprocessors import get_preprocessor, Preprocessor
from ray.rllib.models.utils import get_filter_config
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import ViewRequirementsDict
from ray.rllib.utils.annotations import (
@classmethod
def _get_dist_cls_from_action_space(cls, action_space: gym.Space, *, framework: Optional[str]=None) -> Distribution:
    """Returns a distribution class for the given action space.

        You can get the required input dimension for the distribution by calling
        `action_dict_cls.required_input_dim(action_space)`
        on the retrieved class. This is useful, because the Catalog needs to find out
        about the required input dimension for the distribution before the model that
        outputs these inputs is configured.

        Args:
            action_space: Action space of the target gym env.
            framework: The framework to use.

        Returns:
            The distribution class for the given action space.
        """

    class DistEnum(enum.Enum):
        Categorical = 'Categorical'
        DiagGaussian = 'Gaussian'
        Deterministic = 'Deterministic'
        MultiDistribution = 'MultiDistribution'
        MultiCategorical = 'MultiCategorical'
    if framework == 'torch':
        from ray.rllib.models.torch.torch_distributions import TorchCategorical, TorchDeterministic, TorchDiagGaussian
        distribution_dicts = {DistEnum.Deterministic: TorchDeterministic, DistEnum.DiagGaussian: TorchDiagGaussian, DistEnum.Categorical: TorchCategorical}
    elif framework == 'tf2':
        from ray.rllib.models.tf.tf_distributions import TfCategorical, TfDeterministic, TfDiagGaussian
        distribution_dicts = {DistEnum.Deterministic: TfDeterministic, DistEnum.DiagGaussian: TfDiagGaussian, DistEnum.Categorical: TfCategorical}
    else:
        raise ValueError(f"Unknown framework: {framework}. Only 'torch' and 'tf2' are supported for RLModule Catalogs.")
    if isinstance(action_space, (Tuple, Dict)):
        partial_multi_action_distribution_cls = _multi_action_dist_partial_helper(catalog_cls=cls, action_space=action_space, framework=framework)
        distribution_dicts[DistEnum.MultiDistribution] = partial_multi_action_distribution_cls
    if isinstance(action_space, MultiDiscrete):
        partial_multi_categorical_distribution_cls = _multi_categorical_dist_partial_helper(action_space=action_space, framework=framework)
        distribution_dicts[DistEnum.MultiCategorical] = partial_multi_categorical_distribution_cls
    if isinstance(action_space, Box):
        if action_space.dtype.char in np.typecodes['AllInteger']:
            raise ValueError('Box(..., `int`) action spaces are not supported. Use MultiDiscrete  or Box(..., `float`).')
        else:
            if len(action_space.shape) > 1:
                raise UnsupportedSpaceException(f'Action space has multiple dimensions {action_space.shape}. Consider reshaping this into a single dimension, using a custom action distribution, using a Tuple action space, or the multi-agent API.')
            return distribution_dicts[DistEnum.DiagGaussian]
    elif isinstance(action_space, Discrete):
        return distribution_dicts[DistEnum.Categorical]
    elif isinstance(action_space, (Tuple, Dict)):
        return distribution_dicts[DistEnum.MultiDistribution]
    elif isinstance(action_space, Simplex):
        raise NotImplementedError('Simplex action space not yet supported.')
    elif isinstance(action_space, MultiDiscrete):
        return distribution_dicts[DistEnum.MultiCategorical]
    else:
        raise NotImplementedError(f'Unsupported action space: `{action_space}`')