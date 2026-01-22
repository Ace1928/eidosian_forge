import re
import xml
import xml.etree.ElementTree as ET
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Union
from xml.etree.ElementTree import TreeBuilder
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_core.runnables.utils import AddableDict
class _StreamingParser:
    """Streaming parser for XML.

    This implementation is pulled into a class to avoid implementation
    drift between transform and atransform of the XMLOutputParser.
    """

    def __init__(self, parser: Literal['defusedxml', 'xml']) -> None:
        """Initialize the streaming parser.

        Args:
            parser: Parser to use for XML parsing. Can be either 'defusedxml' or 'xml'.
              See documentation in XMLOutputParser for more information.
        """
        if parser == 'defusedxml':
            try:
                from defusedxml import ElementTree as DET
            except ImportError:
                raise ImportError('defusedxml is not installed. Please install it to use the defusedxml parser.You can install it with `pip install defusedxml` ')
            _parser = DET.DefusedXMLParser(target=TreeBuilder())
        else:
            _parser = None
        self.pull_parser = ET.XMLPullParser(['start', 'end'], _parser=_parser)
        self.xml_start_re = re.compile('<[a-zA-Z:_]')
        self.current_path: List[str] = []
        self.current_path_has_children = False
        self.buffer = ''
        self.xml_started = False

    def parse(self, chunk: Union[str, BaseMessage]) -> Iterator[AddableDict]:
        """Parse a chunk of text.

        Args:
            chunk: A chunk of text to parse. This can be a string or a BaseMessage.

        Yields:
            AddableDict: A dictionary representing the parsed XML element.
        """
        if isinstance(chunk, BaseMessage):
            chunk_content = chunk.content
            if not isinstance(chunk_content, str):
                return
            chunk = chunk_content
        self.buffer += chunk
        if not self.xml_started:
            if (match := self.xml_start_re.search(self.buffer)):
                self.buffer = self.buffer[match.start():]
                self.xml_started = True
            else:
                return
        self.pull_parser.feed(self.buffer)
        self.buffer = ''
        try:
            for event, elem in self.pull_parser.read_events():
                if event == 'start':
                    self.current_path.append(elem.tag)
                    self.current_path_has_children = False
                elif event == 'end':
                    self.current_path.pop()
                    if not self.current_path_has_children:
                        yield nested_element(self.current_path, elem)
                    if self.current_path:
                        self.current_path_has_children = True
                    else:
                        self.xml_started = False
        except xml.etree.ElementTree.ParseError:
            if not self.current_path:
                return
            else:
                raise

    def close(self) -> None:
        """Close the parser."""
        try:
            self.pull_parser.close()
        except xml.etree.ElementTree.ParseError:
            pass