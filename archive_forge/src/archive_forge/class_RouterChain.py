from __future__ import annotations
from abc import ABC
from typing import Any, Dict, List, Mapping, NamedTuple, Optional
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import Extra
from langchain.chains.base import Chain
class RouterChain(Chain, ABC):
    """Chain that outputs the name of a destination chain and the inputs to it."""

    @property
    def output_keys(self) -> List[str]:
        return ['destination', 'next_inputs']

    def route(self, inputs: Dict[str, Any], callbacks: Callbacks=None) -> Route:
        """
        Route inputs to a destination chain.

        Args:
            inputs: inputs to the chain
            callbacks: callbacks to use for the chain

        Returns:
            a Route object
        """
        result = self(inputs, callbacks=callbacks)
        return Route(result['destination'], result['next_inputs'])

    async def aroute(self, inputs: Dict[str, Any], callbacks: Callbacks=None) -> Route:
        result = await self.acall(inputs, callbacks=callbacks)
        return Route(result['destination'], result['next_inputs'])