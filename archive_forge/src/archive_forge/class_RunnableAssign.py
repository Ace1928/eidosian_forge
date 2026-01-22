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
class RunnableAssign(RunnableSerializable[Dict[str, Any], Dict[str, Any]]):
    """
    A runnable that assigns key-value pairs to Dict[str, Any] inputs.

    The `RunnableAssign` class takes input dictionaries and, through a
    `RunnableParallel` instance, applies transformations, then combines
    these with the original data, introducing new key-value pairs based
    on the mapper's logic.

    Examples:
        .. code-block:: python

            # This is a RunnableAssign
            from typing import Dict
            from langchain_core.runnables.passthrough import (
                RunnableAssign,
                RunnableParallel,
            )
            from langchain_core.runnables.base import RunnableLambda

            def add_ten(x: Dict[str, int]) -> Dict[str, int]:
                return {"added": x["input"] + 10}

            mapper = RunnableParallel(
                {"add_step": RunnableLambda(add_ten),}
            )

            runnable_assign = RunnableAssign(mapper)

            # Synchronous example
            runnable_assign.invoke({"input": 5})
            # returns {'input': 5, 'add_step': {'added': 15}}

            # Asynchronous example
            await runnable_assign.ainvoke({"input": 5})
            # returns {'input': 5, 'add_step': {'added': 15}}
    """
    mapper: RunnableParallel[Dict[str, Any]]

    def __init__(self, mapper: RunnableParallel[Dict[str, Any]], **kwargs: Any) -> None:
        super().__init__(mapper=mapper, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'runnable']

    def get_name(self, suffix: Optional[str]=None, *, name: Optional[str]=None) -> str:
        name = name or self.name or f'RunnableAssign<{','.join(self.mapper.steps__.keys())}>'
        return super().get_name(suffix, name=name)

    def get_input_schema(self, config: Optional[RunnableConfig]=None) -> Type[BaseModel]:
        map_input_schema = self.mapper.get_input_schema(config)
        if not map_input_schema.__custom_root_type__:
            return map_input_schema
        return super().get_input_schema(config)

    def get_output_schema(self, config: Optional[RunnableConfig]=None) -> Type[BaseModel]:
        map_input_schema = self.mapper.get_input_schema(config)
        map_output_schema = self.mapper.get_output_schema(config)
        if not map_input_schema.__custom_root_type__ and (not map_output_schema.__custom_root_type__):
            return create_model('RunnableAssignOutput', **{k: (v.type_, v.default) for s in (map_input_schema, map_output_schema) for k, v in s.__fields__.items()})
        elif not map_output_schema.__custom_root_type__:
            return map_output_schema
        return super().get_output_schema(config)

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        return self.mapper.config_specs

    def get_graph(self, config: RunnableConfig | None=None) -> Graph:
        graph = self.mapper.get_graph(config)
        input_node = graph.first_node()
        output_node = graph.last_node()
        if input_node is not None and output_node is not None:
            passthrough_node = graph.add_node(_graph_passthrough)
            graph.add_edge(input_node, passthrough_node)
            graph.add_edge(passthrough_node, output_node)
        return graph

    def _invoke(self, input: Dict[str, Any], run_manager: CallbackManagerForChainRun, config: RunnableConfig, **kwargs: Any) -> Dict[str, Any]:
        assert isinstance(input, dict), 'The input to RunnablePassthrough.assign() must be a dict.'
        return {**input, **self.mapper.invoke(input, patch_config(config, callbacks=run_manager.get_child()), **kwargs)}

    def invoke(self, input: Dict[str, Any], config: Optional[RunnableConfig]=None, **kwargs: Any) -> Dict[str, Any]:
        return self._call_with_config(self._invoke, input, config, **kwargs)

    async def _ainvoke(self, input: Dict[str, Any], run_manager: AsyncCallbackManagerForChainRun, config: RunnableConfig, **kwargs: Any) -> Dict[str, Any]:
        assert isinstance(input, dict), 'The input to RunnablePassthrough.assign() must be a dict.'
        return {**input, **await self.mapper.ainvoke(input, patch_config(config, callbacks=run_manager.get_child()), **kwargs)}

    async def ainvoke(self, input: Dict[str, Any], config: Optional[RunnableConfig]=None, **kwargs: Any) -> Dict[str, Any]:
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    def _transform(self, input: Iterator[Dict[str, Any]], run_manager: CallbackManagerForChainRun, config: RunnableConfig, **kwargs: Any) -> Iterator[Dict[str, Any]]:
        mapper_keys = set(self.mapper.steps__.keys())
        for_passthrough, for_map = safetee(input, 2, lock=threading.Lock())
        map_output = self.mapper.transform(for_map, patch_config(config, callbacks=run_manager.get_child()), **kwargs)
        with get_executor_for_config(config) as executor:
            first_map_chunk_future = executor.submit(next, map_output, None)
            for chunk in for_passthrough:
                assert isinstance(chunk, dict), 'The input to RunnablePassthrough.assign() must be a dict.'
                filtered = AddableDict({k: v for k, v in chunk.items() if k not in mapper_keys})
                if filtered:
                    yield filtered
            yield cast(Dict[str, Any], first_map_chunk_future.result())
            for chunk in map_output:
                yield chunk

    def transform(self, input: Iterator[Dict[str, Any]], config: Optional[RunnableConfig]=None, **kwargs: Any | None) -> Iterator[Dict[str, Any]]:
        yield from self._transform_stream_with_config(input, self._transform, config, **kwargs)

    async def _atransform(self, input: AsyncIterator[Dict[str, Any]], run_manager: AsyncCallbackManagerForChainRun, config: RunnableConfig, **kwargs: Any) -> AsyncIterator[Dict[str, Any]]:
        mapper_keys = set(self.mapper.steps__.keys())
        for_passthrough, for_map = atee(input, 2, lock=asyncio.Lock())
        map_output = self.mapper.atransform(for_map, patch_config(config, callbacks=run_manager.get_child()), **kwargs)
        first_map_chunk_task: asyncio.Task = asyncio.create_task(py_anext(map_output, None))
        async for chunk in for_passthrough:
            assert isinstance(chunk, dict), 'The input to RunnablePassthrough.assign() must be a dict.'
            filtered = AddableDict({k: v for k, v in chunk.items() if k not in mapper_keys})
            if filtered:
                yield filtered
        yield (await first_map_chunk_task)
        async for chunk in map_output:
            yield chunk

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