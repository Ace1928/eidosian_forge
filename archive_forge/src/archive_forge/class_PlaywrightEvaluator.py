import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
class PlaywrightEvaluator(ABC):
    """Abstract base class for all evaluators.

    Each evaluator should take a page, a browser instance, and a response
    object, process the page as necessary, and return the resulting text.
    """

    @abstractmethod
    def evaluate(self, page: 'Page', browser: 'Browser', response: 'Response') -> str:
        """Synchronously process the page and return the resulting text.

        Args:
            page: The page to process.
            browser: The browser instance.
            response: The response from page.goto().

        Returns:
            text: The text content of the page.
        """
        pass

    @abstractmethod
    async def evaluate_async(self, page: 'AsyncPage', browser: 'AsyncBrowser', response: 'AsyncResponse') -> str:
        """Asynchronously process the page and return the resulting text.

        Args:
            page: The page to process.
            browser: The browser instance.
            response: The response from page.goto().

        Returns:
            text: The text content of the page.
        """
        pass