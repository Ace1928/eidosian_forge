import logging
import sys
from collections import defaultdict
from typing import (
from uuid import UUID
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult
Run when Retriever ends running.