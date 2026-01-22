from typing import Any, Dict, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools.amadeus.base import AmadeusBaseTool
class AmadeusClosestAirport(AmadeusBaseTool):
    """Tool for finding the closest airport to a particular location."""
    name: str = 'closest_airport'
    description: str = 'Use this tool to find the closest airport to a particular location.'
    args_schema: Type[ClosestAirportSchema] = ClosestAirportSchema
    llm: Optional[BaseLanguageModel] = Field(default=None)
    "Tool's llm used for calculating the closest airport. Defaults to `ChatOpenAI`."

    @root_validator(pre=True)
    def set_llm(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not values.get('llm'):
            values['llm'] = ChatOpenAI(temperature=0)
        return values

    def _run(self, location: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        content = f""" What is the nearest airport to {location}? Please respond with the  airport's International Air Transport Association (IATA) Location  Identifier in the following JSON format. JSON: "iataCode": "IATA  Location Identifier" """
        return self.llm.invoke(content)