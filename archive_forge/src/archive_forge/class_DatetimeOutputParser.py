import random
from datetime import datetime, timedelta
from typing import List
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.utils import comma_list
class DatetimeOutputParser(BaseOutputParser[datetime]):
    """Parse the output of an LLM call to a datetime."""
    format: str = '%Y-%m-%dT%H:%M:%S.%fZ'
    'The string value that used as the datetime format.'

    def get_format_instructions(self) -> str:
        examples = comma_list(_generate_random_datetime_strings(self.format))
        return f"Write a datetime string that matches the following pattern: '{self.format}'.\n\nExamples: {examples}\n\nReturn ONLY this string, no other words!"

    def parse(self, response: str) -> datetime:
        try:
            return datetime.strptime(response.strip(), self.format)
        except ValueError as e:
            raise OutputParserException(f'Could not parse datetime string: {response}') from e

    @property
    def _type(self) -> str:
        return 'datetime'