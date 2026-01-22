from __future__ import annotations
import asyncio
import inspect
import threading
from typing import (
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import (
from langchain_core.runnables.config import (
from langchain_core.runnables.graph import Graph
from langchain_core.runnables.utils import (
from langchain_core.utils.aiter import atee, py_anext
from langchain_core.utils.iter import safetee
class RunnablePick(RunnableSerializable[Dict[str, Any], Dict[str, Any]]):
    """Runnable that picks keys from Dict[str, Any] inputs.

    RunnablePick class represents a runnable that selectively picks keys from a
    dictionary input. It allows you to specify one or more keys to extract
    from the input dictionary. It returns a new dictionary containing only
    the selected keys.

    Example :
        .. code-block:: python

            from langchain_core.runnables.passthrough import RunnablePick

            input_data = {
                'name': 'John',
                'age': 30,
                'city': 'New York',
                'country': 'USA'
            }

            runnable = RunnablePick(keys=['name', 'age'])

            output_data = runnable.invoke(input_data)

            print(output_data)  # Output: {'name': 'John', 'age': 30}
    """
    keys: Union[str, List[str]]

    def __init__(self, keys: Union[str, List[str]], **kwargs: Any) -> None:
        super().__init__(keys=keys, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'runnable']

    def get_name(self, suffix: Optional[str]=None, *, name: Optional[str]=None) -> str:
        name = name or self.name or f'RunnablePick<{','.join([self.keys] if isinstance(self.keys, str) else self.keys)}>'
        return super().get_name(suffix, name=name)

    def _pick(self, input: Dict[str, Any]) -> Any:
        assert isinstance(input, dict), 'The input to RunnablePassthrough.assign() must be a dict.'
        if isinstance(self.keys, str):
            return input.get(self.keys)
        else:
            picked = {k: input.get(k) for k in self.keys if k in input}
            if picked:
                return AddableDict(picked)
            else:
                return None

    def _invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self._pick(input)

    def invoke(self, input: Dict[str, Any], config: Optional[RunnableConfig]=None, **kwargs: Any) -> Dict[str, Any]:
        return self._call_with_config(self._invoke, input, config, **kwargs)

    async def _ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self._pick(input)

    async def ainvoke(self, input: Dict[str, Any], config: Optional[RunnableConfig]=None, **kwargs: Any) -> Dict[str, Any]:
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    def _transform(self, input: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        for chunk in input:
            picked = self._pick(chunk)
            if picked is not None:
                yield picked

    def transform(self, input: Iterator[Dict[str, Any]], config: Optional[RunnableConfig]=None, **kwargs: Any) -> Iterator[Dict[str, Any]]:
        yield from self._transform_stream_with_config(input, self._transform, config, **kwargs)

    async def _atransform(self, input: AsyncIterator[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
        async for chunk in input:
            picked = self._pick(chunk)
            if picked is not None:
                yield picked

    async def atransform(self, input: AsyncIterator[Dict[str, Any]], config: Optional[RunnableConfig]=None, **kwargs: Any) -> AsyncIterator[Dict[str, Any]]:
        async for chunk in self._atransform_stream_with_config(input, self._atransform, config, **kwargs):
            yield chunk

    def stream(self, input: Dict[str, Any], config: Optional[RunnableConfig]=None, **kwargs: Any) -> Iterator[Dict[str, Any]]:
        return self.transform(iter([input]), config, **kwargs)

    async def astream(self, input: Dict[str, Any], config: Optional[RunnableConfig]=None, **kwargs: Any) -> AsyncIterator[Dict[str, Any]]:

        async def input_aiter() -> AsyncIterator[Dict[str, Any]]:
            yield input
        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk