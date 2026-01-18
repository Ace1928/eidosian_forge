from typing import Any, Dict, List, Optional
from langchain_core._api.deprecation import deprecated
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
Run query through GoogleSearch and return metadata.

        Args:
            query: The query to search for.
            num_results: The number of results to return.
            search_params: Parameters to be passed on search

        Returns:
            A list of dictionaries with the following keys:
                snippet - The description of the result.
                title - The title of the result.
                link - The link to the result.
        