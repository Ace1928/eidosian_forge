from typing import Any, Dict, Optional, Sequence, Type, Union
from sqlalchemy.engine import Result
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import (
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import BaseTool
from langchain_community.tools.sql_database.prompt import QUERY_CHECKER
class ListSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for getting tables names."""
    name: str = 'sql_db_list_tables'
    description: str = 'Input is an empty string, output is a comma-separated list of tables in the database.'
    args_schema: Type[BaseModel] = _ListSQLDataBaseToolInput

    def _run(self, tool_input: str='', run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        """Get a comma-separated list of table names."""
        return ', '.join(self.db.get_usable_table_names())