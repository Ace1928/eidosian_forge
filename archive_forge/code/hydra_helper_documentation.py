import logging
from hydra.core.config_store import ConfigStore
from omegaconf.errors import ValidationError
from xformers.components.attention import ATTENTION_REGISTRY
from xformers.components.feedforward import FEEDFORWARD_REGISTRY
from xformers.components.positional_embedding import POSITION_EMBEDDING_REGISTRY

    Best effort - OmegaConf supports limited typing, so we may fail to import
    certain config classes. For example, pytorch typing are not supported.
    