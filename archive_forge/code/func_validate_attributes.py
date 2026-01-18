from typing import List
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools.connery import ConneryService
@root_validator()
def validate_attributes(cls, values: dict) -> dict:
    """
        Validate the attributes of the ConneryToolkit class.
        Parameters:
            values (dict): The arguments to validate.
        Returns:
            dict: The validated arguments.
        """
    if not values.get('tools'):
        raise ValueError("The attribute 'tools' must be set.")
    return values