import json
import logging
import os
import re
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.output_parsers.transform import BaseOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult, Generation
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
class KineticaSqlOutputParser(BaseOutputParser[KineticaSqlResponse]):
    """Fetch and return data from the Kinetica LLM.

    This object is used as the last element of a chain to execute generated SQL and it
    will output a ``KineticaSqlResponse`` containing the SQL and a pandas dataframe with
    the fetched data.

    Example:
        .. code-block:: python

            from langchain_community.chat_models.kinetica import (
                KineticaChatLLM, KineticaSqlOutputParser)
            kinetica_llm = KineticaChatLLM()

            # create chain
            ctx_messages = kinetica_llm.load_messages_from_context(self.context_name)
            ctx_messages.append(("human", "{input}"))
            prompt_template = ChatPromptTemplate.from_messages(ctx_messages)
            chain = (
                prompt_template
                | kinetica_llm
                | KineticaSqlOutputParser(kdbc=kinetica_llm.kdbc)
            )
            sql_response: KineticaSqlResponse = chain.invoke(
                {"input": "What are the female users ordered by username?"}
            )

            assert isinstance(sql_response, KineticaSqlResponse)
            LOG.info(f"SQL Response: {sql_response.sql}")
            assert isinstance(sql_response.dataframe, pd.DataFrame)
    """
    kdbc: Any = Field(exclude=True)
    ' Kinetica DB connection. '

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def parse(self, text: str) -> KineticaSqlResponse:
        df = self.kdbc.to_df(text)
        return KineticaSqlResponse(sql=text, dataframe=df)

    def parse_result(self, result: List[Generation], *, partial: bool=False) -> KineticaSqlResponse:
        return self.parse(result[0].text)

    @property
    def _type(self) -> str:
        return 'kinetica_sql_output_parser'