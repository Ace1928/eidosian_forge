import pytorch_lightning as pl
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _graphcore_available_and_importable
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
def __verify_configure_model_configuration(model: 'pl.LightningModule') -> None:
    if is_overridden('configure_sharded_model', model):
        name = type(model).__name__
        if is_overridden('configure_model', model):
            raise RuntimeError(f'Both `{name}.configure_model`, and `{name}.configure_sharded_model` are overridden. The latter is deprecated and it should be replaced with the former.')
        rank_zero_deprecation(f'You have overridden `{name}.configure_sharded_model` which is deprecated. Please override the `configure_model` hook instead. Instantiation with the newer hook will be created on the device right away and have the right data type depending on the precision setting in the Trainer.')