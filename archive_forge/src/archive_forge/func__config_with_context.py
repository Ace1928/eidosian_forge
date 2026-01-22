import asyncio
import threading
from collections import defaultdict
from functools import partial
from itertools import groupby
from typing import (
from langchain_core._api.beta_decorator import beta
from langchain_core.runnables.base import (
from langchain_core.runnables.config import RunnableConfig, ensure_config, patch_config
from langchain_core.runnables.utils import ConfigurableFieldSpec, Input, Output
def _config_with_context(config: RunnableConfig, steps: List[Runnable], setter: Callable, getter: Callable, event_cls: Union[Type[threading.Event], Type[asyncio.Event]]) -> RunnableConfig:
    if any((k.startswith(CONTEXT_CONFIG_PREFIX) for k in config.get('configurable', {}))):
        return config
    context_specs = [(spec, i) for i, step in enumerate(steps) for spec in step.config_specs if spec.id.startswith(CONTEXT_CONFIG_PREFIX)]
    grouped_by_key = {key: list(group) for key, group in groupby(sorted(context_specs, key=lambda s: s[0].id), key=lambda s: _key_from_id(s[0].id))}
    deps_by_key = {key: set((_key_from_id(dep) for spec in group for dep in spec[0].dependencies or [])) for key, group in grouped_by_key.items()}
    values: Values = {}
    events: DefaultDict[str, Union[asyncio.Event, threading.Event]] = defaultdict(event_cls)
    context_funcs: Dict[str, Callable[[], Any]] = {}
    for key, group in grouped_by_key.items():
        getters = [s for s in group if s[0].id.endswith(CONTEXT_CONFIG_SUFFIX_GET)]
        setters = [s for s in group if s[0].id.endswith(CONTEXT_CONFIG_SUFFIX_SET)]
        for dep in deps_by_key[key]:
            if key in deps_by_key[dep]:
                raise ValueError(f'Deadlock detected between context keys {key} and {dep}')
        if len(setters) != 1:
            raise ValueError(f'Expected exactly one setter for context key {key}')
        setter_idx = setters[0][1]
        if any((getter_idx < setter_idx for _, getter_idx in getters)):
            raise ValueError(f'Context setter for key {key} must be defined after all getters.')
        if getters:
            context_funcs[getters[0][0].id] = partial(getter, events[key], values)
        context_funcs[setters[0][0].id] = partial(setter, events[key], values)
    return patch_config(config, configurable=context_funcs)