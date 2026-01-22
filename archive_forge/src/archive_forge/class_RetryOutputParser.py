from __future__ import annotations
from typing import Any, TypeVar
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
class RetryOutputParser(BaseOutputParser[T]):
    """Wraps a parser and tries to fix parsing errors.

    Does this by passing the original prompt and the completion to another
    LLM, and telling it the completion did not satisfy criteria in the prompt.
    """
    parser: BaseOutputParser[T]
    'The parser to use to parse the output.'
    retry_chain: Any
    'The LLMChain to use to retry the completion.'
    max_retries: int = 1
    'The maximum number of times to retry the parse.'

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, parser: BaseOutputParser[T], prompt: BasePromptTemplate=NAIVE_RETRY_PROMPT, max_retries: int=1) -> RetryOutputParser[T]:
        """Create an RetryOutputParser from a language model and a parser.

        Args:
            llm: llm to use for fixing
            parser: parser to use for parsing
            prompt: prompt to use for fixing
            max_retries: Maximum number of retries to parse.

        Returns:
            RetryOutputParser
        """
        from langchain.chains.llm import LLMChain
        chain = LLMChain(llm=llm, prompt=prompt)
        return cls(parser=parser, retry_chain=chain, max_retries=max_retries)

    def parse_with_prompt(self, completion: str, prompt_value: PromptValue) -> T:
        """Parse the output of an LLM call using a wrapped parser.

        Args:
            completion: The chain completion to parse.
            prompt_value: The prompt to use to parse the completion.

        Returns:
            The parsed completion.
        """
        retries = 0
        while retries <= self.max_retries:
            try:
                return self.parser.parse(completion)
            except OutputParserException as e:
                if retries == self.max_retries:
                    raise e
                else:
                    retries += 1
                    completion = self.retry_chain.run(prompt=prompt_value.to_string(), completion=completion)
        raise OutputParserException('Failed to parse')

    async def aparse_with_prompt(self, completion: str, prompt_value: PromptValue) -> T:
        """Parse the output of an LLM call using a wrapped parser.

        Args:
            completion: The chain completion to parse.
            prompt_value: The prompt to use to parse the completion.

        Returns:
            The parsed completion.
        """
        retries = 0
        while retries <= self.max_retries:
            try:
                return await self.parser.aparse(completion)
            except OutputParserException as e:
                if retries == self.max_retries:
                    raise e
                else:
                    retries += 1
                    completion = await self.retry_chain.arun(prompt=prompt_value.to_string(), completion=completion)
        raise OutputParserException('Failed to parse')

    def parse(self, completion: str) -> T:
        raise NotImplementedError('This OutputParser can only be called by the `parse_with_prompt` method.')

    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()

    @property
    def _type(self) -> str:
        return 'retry'