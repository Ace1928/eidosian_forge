from typing import List
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.cogniswitch.tool import (
class CogniswitchToolkit(BaseToolkit):
    """
    Toolkit for CogniSwitch.

    Use the toolkit to get all the tools present in the cogniswitch and
    use them to interact with your knowledge
    """
    cs_token: str
    OAI_token: str
    apiKey: str

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [CogniswitchKnowledgeStatus(cs_token=self.cs_token, OAI_token=self.OAI_token, apiKey=self.apiKey), CogniswitchKnowledgeRequest(cs_token=self.cs_token, OAI_token=self.OAI_token, apiKey=self.apiKey), CogniswitchKnowledgeSourceFile(cs_token=self.cs_token, OAI_token=self.OAI_token, apiKey=self.apiKey), CogniswitchKnowledgeSourceURL(cs_token=self.cs_token, OAI_token=self.OAI_token, apiKey=self.apiKey)]