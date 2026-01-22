import asyncio
import functools
import inspect
import logging
import sys
from typing import Any, Dict, Optional, Sequence, TypeVar
import wandb.sdk
import wandb.util
from wandb.sdk.lib import telemetry as wb_telemetry
from wandb.sdk.lib.timer import Timer
class AutologAPI:

    def __init__(self, name: str, symbols: Sequence[str], resolver: ArgumentResponseResolver, telemetry_feature: Optional[str]=None) -> None:
        """Autolog API calls to W&B."""
        self._telemetry_feature = telemetry_feature
        self._patch_api = PatchAPI(name=name, symbols=symbols, resolver=resolver)
        self._name = self._patch_api.name
        self._run: Optional[wandb.sdk.wandb_run.Run] = None
        self.__run_created_by_autolog: bool = False

    @property
    def _is_enabled(self) -> bool:
        """Returns whether autologging is enabled."""
        return self._run is not None

    def __call__(self, init: AutologInitArgs=None) -> None:
        """Enable autologging."""
        self.enable(init=init)

    def _run_init(self, init: AutologInitArgs=None) -> None:
        """Handle wandb run initialization."""
        if init:
            _wandb_run = wandb.run
            self._run = wandb.init(**init)
            if _wandb_run != self._run:
                self.__run_created_by_autolog = True
        elif wandb.run is None:
            self._run = wandb.init()
            self.__run_created_by_autolog = True
        else:
            self._run = wandb.run

    def enable(self, init: AutologInitArgs=None) -> None:
        """Enable autologging.

        Args:
            init: Optional dictionary of arguments to pass to wandb.init().

        """
        if self._is_enabled:
            logger.info(f'{self._name} autologging is already enabled, disabling and re-enabling.')
            self.disable()
        logger.info(f'Enabling {self._name} autologging.')
        self._run_init(init=init)
        self._patch_api.patch(self._run)
        if self._telemetry_feature:
            with wb_telemetry.context(self._run) as tel:
                setattr(tel.feature, self._telemetry_feature, True)

    def disable(self) -> None:
        """Disable autologging."""
        if self._run is None:
            return
        logger.info(f'Disabling {self._name} autologging.')
        if self.__run_created_by_autolog:
            self._run.finish()
            self.__run_created_by_autolog = False
        self._run = None
        self._patch_api.unpatch()