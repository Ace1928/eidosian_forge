from typing import Any, Dict, Optional
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import (
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities.spark_sql import SparkSQL
from langchain_core.tools import BaseTool
from langchain_community.tools.spark_sql.prompt import QUERY_CHECKER
class ListSparkSQLTool(BaseSparkSQLTool, BaseTool):
    """Tool for getting tables names."""
    name: str = 'list_tables_sql_db'
    description: str = 'Input is an empty string, output is a comma separated list of tables in the Spark SQL.'

    def _run(self, tool_input: str='', run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        """Get the schema for a specific table."""
        return ', '.join(self.db.get_usable_table_names())