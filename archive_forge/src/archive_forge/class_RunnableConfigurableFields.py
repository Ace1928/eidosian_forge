from __future__ import annotations
import enum
import threading
from abc import abstractmethod
from functools import wraps
from typing import (
from weakref import WeakValueDictionary
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import Runnable, RunnableSerializable
from langchain_core.runnables.config import (
from langchain_core.runnables.graph import Graph
from langchain_core.runnables.utils import (
class RunnableConfigurableFields(DynamicRunnable[Input, Output]):
    """Runnable that can be dynamically configured.

    A RunnableConfigurableFields should be initiated using the
    `configurable_fields` method of a Runnable.

    Here is an example of using a RunnableConfigurableFields with LLMs:

        .. code-block:: python

            from langchain_core.prompts import PromptTemplate
            from langchain_core.runnables import ConfigurableField
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(temperature=0).configurable_fields(
                temperature=ConfigurableField(
                    id="temperature",
                    name="LLM Temperature",
                    description="The temperature of the LLM",
                )
            )
            # This creates a RunnableConfigurableFields for a chat model.

            # When invoking the created RunnableSequence, you can pass in the
            # value for your ConfigurableField's id which in this case
            # will be change in temperature

            prompt = PromptTemplate.from_template("Pick a random number above {x}")
            chain = prompt | model

            chain.invoke({"x": 0})
            chain.invoke({"x": 0}, config={"configurable": {"temperature": 0.9}})


    Here is an example of using a RunnableConfigurableFields with HubRunnables:

        .. code-block:: python

            from langchain_core.prompts import PromptTemplate
            from langchain_core.runnables import ConfigurableField
            from langchain_openai import ChatOpenAI
            from langchain.runnables.hub import HubRunnable

            prompt = HubRunnable("rlm/rag-prompt").configurable_fields(
                owner_repo_commit=ConfigurableField(
                    id="hub_commit",
                    name="Hub Commit",
                    description="The Hub commit to pull from",
                )
            )

            prompt.invoke({"question": "foo", "context": "bar"})

            # Invoking prompt with `with_config` method

            prompt.invoke(
                {"question": "foo", "context": "bar"},
                config={"configurable": {"hub_commit": "rlm/rag-prompt-llama"}},
            )
    """
    fields: Dict[str, AnyConfigurableField]

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'runnable']

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        return get_unique_config_specs([ConfigurableFieldSpec(id=spec.id, name=spec.name, description=spec.description or self.default.__fields__[field_name].field_info.description, annotation=spec.annotation or self.default.__fields__[field_name].annotation, default=getattr(self.default, field_name), is_shared=spec.is_shared) if isinstance(spec, ConfigurableField) else make_options_spec(spec, self.default.__fields__[field_name].field_info.description) for field_name, spec in self.fields.items()] + list(self.default.config_specs))

    def configurable_fields(self, **kwargs: AnyConfigurableField) -> RunnableSerializable[Input, Output]:
        return self.default.configurable_fields(**{**self.fields, **kwargs})

    def _prepare(self, config: Optional[RunnableConfig]=None) -> Tuple[Runnable[Input, Output], RunnableConfig]:
        config = ensure_config(config)
        specs_by_id = {spec.id: (key, spec) for key, spec in self.fields.items()}
        configurable_fields = {specs_by_id[k][0]: v for k, v in config.get('configurable', {}).items() if k in specs_by_id and isinstance(specs_by_id[k][1], ConfigurableField)}
        configurable_single_options = {k: v.options[config.get('configurable', {}).get(v.id) or v.default] for k, v in self.fields.items() if isinstance(v, ConfigurableFieldSingleOption)}
        configurable_multi_options = {k: [v.options[o] for o in config.get('configurable', {}).get(v.id, v.default)] for k, v in self.fields.items() if isinstance(v, ConfigurableFieldMultiOption)}
        configurable = {**configurable_fields, **configurable_single_options, **configurable_multi_options}
        if configurable:
            init_params = {k: v for k, v in self.default.__dict__.items() if k in self.default.__fields__}
            return (self.default.__class__(**{**init_params, **configurable}), config)
        else:
            return (self.default, config)