from __future__ import annotations
import logging
import re
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Optional, Union
import torch
from accelerate.hooks import AlignDevicesHook
from accelerate.utils import named_module_tensors, offload_state_dict
from torch import nn
from transformers import PreTrainedModel
from transformers.pytorch_utils import Conv1D
from peft.utils import INCLUDE_LINEAR_LAYERS_SHORTHAND
from ..config import PeftConfig
from ..utils import ModulesToSaveWrapper, _get_submodules
class BaseTuner(nn.Module, ABC):
    """
    A base tuner model that provides the common methods and attributes for all tuners that are injectable into a
    torch.nn.Module

    For adding a new Tuner class, one needs to overwrite the following methods:

    - **_prepare_adapter_config**:
        A private method to eventually prepare the adapter config, for example in case the field `target_modules` is
        missing.
    - **_create_and_replace**:
        A private method to create and replace the target module with the adapter module.
    - **_check_target_module_exists**:
        A private helper method to check if the passed module's key name matches any of the target modules in the
        adapter_config.

    The easiest is to check what is done in the `peft.tuners.lora.LoraModel` class.

    Attributes:
        model (`torch.nn.Module`):
            The model to which the adapter tuner layers will be attached.
        forward (`Callable`):
            The forward method of the model.
        peft_config (`Union[`PeftConfig`, dict[str, PeftConfig]]`):
            The adapter configuration object, it should be a dictionary of `str` to `PeftConfig` objects. One can also
            pass a PeftConfig object and a new adapter will be created with the default name `adapter` or create a new
            dictionary with a key `adapter_name` and a value of that peft config.
        config (`dict[str, Any]`):
            The model configuration object, it should be a dictionary of `str` to `Any` objects.
        targeted_module_names (`list[str]`):
            The list of module names that were actually adapted. Can be useful to inspect if you want to quickly
            double-check that the `config.target_modules` where specified correctly.
    """

    def __init__(self, model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str) -> None:
        super().__init__()
        self.model = model
        self.targeted_module_names: list[str] = []
        if not hasattr(self, 'peft_config'):
            self.peft_config = {adapter_name: peft_config} if isinstance(peft_config, PeftConfig) else peft_config
        else:
            logger.info('Already found a `peft_config` attribute in the model. This will lead to having multiple adapters in the model. Make sure to know what you are doing!')
            if isinstance(peft_config, PeftConfig):
                self.peft_config[adapter_name] = peft_config
            else:
                self.peft_config.update(peft_config)
        self.active_adapter = adapter_name
        self.inject_adapter(self.model, adapter_name)
        self.model.peft_config = self.peft_config

    @property
    def active_adapters(self) -> list[str]:
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        return self.active_adapter

    def forward(self, *args: Any, **kwargs: Any):
        return self.model.forward(*args, **kwargs)

    @abstractmethod
    def _prepare_adapter_config(self, peft_config: PeftConfig, model_config: dict) -> PeftConfig:
        """
        A private method to eventually prepare the adapter config. For transformers based models, if
        `peft_config.target_modules` is None, we can automatically infer the target modules from the
        `TRANSFORMERS_MODELS_TO_XXX_TARGET_MODULES_MAPPING`. This method can be further refactored in the future to
        automatically infer it for all tuner models.

        Check out `peft.tuner.lora.LoraModel._prepare_adapter_config` for an example.

        Args:
            peft_config (`str`):
                The adapter config.
            model_config (`str`):
                The transformers model config, that config should contain the `model_type` key.
        """
        ...

    @abstractmethod
    def _check_target_module_exists(peft_config: PeftConfig, key: str) -> bool:
        """
        A helper private method to check if the passed module's key name matches any of the target modules in the
        `peft_config.target_modules` list. If it does, return `True`, else return `False`.

        Args:
            peft_config (`PeftConfig`):
                The adapter config.
            key (`str`):
                The module's key name.
        """
        ...

    @abstractmethod
    def _create_and_replace(self, peft_config: PeftConfig, adapter_name: str, target: nn.Module, target_name: str, parent: nn.Module, current_key: str) -> None:
        """
        Inplace replacement of the target module with the adapter layer. This method needs to be overridden by all the
        tuner classes.

        Check `peft.tuners.lora.LoraModel._create_and_replace` for an example.

        Args:
            peft_config (`PeftConfig`):
                The adapter config.
            adapter_name (`str`):
                The adapter name.
            target (`nn.Module`):
                The target module.
            target_name (`str`):
                The target module's name.
            parent (`nn.Module`):
                The parent module.
            current_key (`str`):
                The key of the current target being adapted.
        """
        ...

    @abstractmethod
    def _mark_only_adapters_as_trainable(self, model: nn.Module):
        """
        A helper method to mark only the adapter layers as trainable (i.e. module.requires_grad = False) This needs to
        be overridden for all tuner classes to match the correct key names.

        Check `peft.tuners.lora.LoraModel._mark_only_adapters_as_trainable` for an example.
        """
        ...

    def _check_new_adapter_config(self, config: PeftConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        pass

    def inject_adapter(self, model: nn.Module, adapter_name: str):
        """
        Creates adapter layers and replaces the target modules with the adapter layers. This method is called under the
        hood by `peft.mapping.get_peft_model` if a non-prompt tuning adapter class is passed.

        The corresponding PEFT config is directly retrieved from the `peft_config` attribute of the BaseTuner class.

        Args:
            model (`nn.Module`):
                The model to be tuned.
            adapter_name (`str`):
                The adapter name.
        """
        peft_config = self.peft_config[adapter_name]
        self._check_new_adapter_config(peft_config)
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in model.named_modules()]
        _check_for_modules_to_save = getattr(peft_config, 'modules_to_save', None) is not None
        _has_modules_to_save = False
        model_config = getattr(model, 'config', {'model_type': 'custom'})
        if hasattr(model_config, 'to_dict'):
            model_config = model_config.to_dict()
        peft_config = self._prepare_adapter_config(peft_config, model_config)
        peft_config = _maybe_include_all_linear_layers(peft_config, model)
        for key in key_list:
            if _check_for_modules_to_save and any((key.endswith(f'{module_to_save}') for module_to_save in peft_config.modules_to_save)):
                parent, target, target_name = _get_submodules(model, key)
                if not isinstance(target, ModulesToSaveWrapper):
                    new_module = ModulesToSaveWrapper(target, adapter_name)
                    setattr(parent, target_name, new_module)
                else:
                    target.update(adapter_name)
                _has_modules_to_save = True
                continue
            if not self._check_target_module_exists(peft_config, key):
                continue
            self.targeted_module_names.append(key)
            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(model, key)
            self._create_and_replace(peft_config, adapter_name, target, target_name, parent, current_key=key)
        if not is_target_modules_in_base_model:
            raise ValueError(f'Target modules {peft_config.target_modules} not found in the base model. Please check the target modules and try again.')
        self._mark_only_adapters_as_trainable(model)
        if self.peft_config[adapter_name].inference_mode:
            for n, p in model.named_parameters():
                if adapter_name in n:
                    p.requires_grad = False
        if _has_modules_to_save:
            if not hasattr(model, 'modules_to_save'):
                model.modules_to_save = set(peft_config.modules_to_save)
            else:
                model.modules_to_save.update(set(peft_config.modules_to_save))

    def merge_adapter(self, adapter_names: Optional[list[str]]=None) -> None:
        """
        This method merges the adapter layers into the base model.

        Merging adapters can lead to a speed up of the forward pass. A copy of the adapter weights is still kept in
        memory, which is required to unmerge the adapters. In order to merge the adapter weights without keeping them
        in memory, please call `merge_and_unload`.

        Args:
            safe_merge (`bool`, *optional*):
                If `True`, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active adapters will be merged.
                Defaults to `None`.
        """
        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                with onload_layer(module):
                    module.merge(adapter_names=adapter_names)

    def unmerge_adapter(self):
        """
        This method unmerges all merged adapter layers from the base model.
        """
        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                with onload_layer(module):
                    module.unmerge()

    def _unloading_checks(self, adapter_names: Optional[list[str]]):
        adapters_to_consider = adapter_names or self.active_adapters
        is_modules_to_save_available = any((self.peft_config[adapter].modules_to_save for adapter in adapters_to_consider))
        if is_modules_to_save_available and len(adapters_to_consider) > 1:
            raise ValueError('Cannot unload multiple adapters that specify `modules_to_save`.')