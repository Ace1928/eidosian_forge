import asyncio
import logging
from typing import List, Optional, Sequence
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain.chains.llm import LLMChain
Get unique Documents.

        Args:
            documents: List of retrieved Documents

        Returns:
            List of unique retrieved Documents
        