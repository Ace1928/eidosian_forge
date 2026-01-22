from dataclasses import dataclass
from string import Template
from typing import List
from enum import Enum
@dataclass
class ConversationTurn:
    message: str
    agent_type: AgentType