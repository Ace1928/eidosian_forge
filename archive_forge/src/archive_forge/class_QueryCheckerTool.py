from typing import Any, Dict, Optional
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import (
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities.spark_sql import SparkSQL
from langchain_core.tools import BaseTool
from langchain_community.tools.spark_sql.prompt import QUERY_CHECKER
class QueryCheckerTool(BaseSparkSQLTool, BaseTool):
    """Use an LLM to check if a query is correct.
    Adapted from https://www.patterns.app/blog/2023/01/18/crunchbot-sql-analyst-gpt/"""
    template: str = QUERY_CHECKER
    llm: BaseLanguageModel
    llm_chain: Any = Field(init=False)
    name: str = 'query_checker_sql_db'
    description: str = '\n    Use this tool to double check if your query is correct before executing it.\n    Always use this tool before executing a query with query_sql_db!\n    '

    @root_validator(pre=True)
    def initialize_llm_chain(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if 'llm_chain' not in values:
            from langchain.chains.llm import LLMChain
            values['llm_chain'] = LLMChain(llm=values.get('llm'), prompt=PromptTemplate(template=QUERY_CHECKER, input_variables=['query']))
        if values['llm_chain'].prompt.input_variables != ['query']:
            raise ValueError("LLM chain for QueryCheckerTool need to use ['query'] as input_variables for the embedded prompt")
        return values

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        """Use the LLM to check the query."""
        return self.llm_chain.predict(query=query, callbacks=run_manager.get_child() if run_manager else None)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun]=None) -> str:
        return await self.llm_chain.apredict(query=query, callbacks=run_manager.get_child() if run_manager else None)