from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Sequence, Union
from langchain_core.messages import (
from langchain_core.runnables import run_in_executor
Return a string representation of the chat history.