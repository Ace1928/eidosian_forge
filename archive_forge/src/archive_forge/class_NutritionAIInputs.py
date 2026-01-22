from typing import Dict, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.passio_nutrition_ai import NutritionAIAPI
class NutritionAIInputs(BaseModel):
    """Inputs to the Passio Nutrition AI tool."""
    query: str = Field(description='A query to look up using Passio Nutrition AI, usually a few words.')