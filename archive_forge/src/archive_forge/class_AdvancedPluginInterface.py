import importlib
import pkgutil
import os
import inspect
import asyncio
from typing import Dict, Type, Any, List, Optional, Union
from types import ModuleType, FunctionType
from abc import ABC, abstractmethod
from dependency_injector import containers, providers
import events  # Assumed robust asynchronous event handling.
import traceback
import json
import logging
from pydantic import BaseModel, create_model, ValidationError
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress
class AdvancedPluginInterface(ABC):
    """
    An abstract base class defining the interface for plugins, encapsulating lifecycle methods,
    event handling, configuration, and health checks to establish a robust plugin architecture.
    """

    @abstractmethod
    async def activate(self) -> None:
        """Asynchronously activate the plugin, initializing necessary resources and operations."""
        pass

    @abstractmethod
    async def deactivate(self) -> None:
        """Asynchronously deactivate the plugin, releasing resources and halting operations cleanly."""
        pass

    @abstractmethod
    def is_active(self) -> bool:
        """Check whether the plugin is active, supporting dynamic plugin management."""
        pass

    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the plugin using a given dictionary, enabling adaptive plugin behavior."""
        pass

    @abstractmethod
    async def on_event(self, event: str, data: Any) -> None:
        """Handle an asynchronous event, allowing plugins to respond to system-wide activities."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Perform an asynchronous health check to evaluate the plugin's operational status."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Retrieve plugin metadata, offering insights into its identity, status, and capabilities."""
        return {'name': self.__class__.__name__, 'module': self.__class__.__module__, 'description': inspect.getdoc(self), 'active': self.is_active()}