from __future__ import annotations
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Iterable, Any
from . import util
class BlockParser:
    """ Parse Markdown blocks into an `ElementTree` object.

    A wrapper class that stitches the various `BlockProcessors` together,
    looping through them and creating an `ElementTree` object.

    """

    def __init__(self, md: Markdown):
        """ Initialize the block parser.

        Arguments:
            md: A Markdown instance.

        Attributes:
            BlockParser.md (Markdown): A Markdown instance.
            BlockParser.state (State): Tracks the nesting level of current location in document being parsed.
            BlockParser.blockprocessors (util.Registry): A collection of
                [`blockprocessors`][markdown.blockprocessors].

        """
        self.blockprocessors: util.Registry[BlockProcessor] = util.Registry()
        self.state = State()
        self.md = md

    def parseDocument(self, lines: Iterable[str]) -> etree.ElementTree:
        """ Parse a Markdown document into an `ElementTree`.

        Given a list of lines, an `ElementTree` object (not just a parent
        `Element`) is created and the root element is passed to the parser
        as the parent. The `ElementTree` object is returned.

        This should only be called on an entire document, not pieces.

        Arguments:
            lines: A list of lines (strings).

        Returns:
            An element tree.
        """
        self.root = etree.Element(self.md.doc_tag)
        self.parseChunk(self.root, '\n'.join(lines))
        return etree.ElementTree(self.root)

    def parseChunk(self, parent: etree.Element, text: str) -> None:
        """ Parse a chunk of Markdown text and attach to given `etree` node.

        While the `text` argument is generally assumed to contain multiple
        blocks which will be split on blank lines, it could contain only one
        block. Generally, this method would be called by extensions when
        block parsing is required.

        The `parent` `etree` Element passed in is altered in place.
        Nothing is returned.

        Arguments:
            parent: The parent element.
            text: The text to parse.

        """
        self.parseBlocks(parent, text.split('\n\n'))

    def parseBlocks(self, parent: etree.Element, blocks: list[str]) -> None:
        """ Process blocks of Markdown text and attach to given `etree` node.

        Given a list of `blocks`, each `blockprocessor` is stepped through
        until there are no blocks left. While an extension could potentially
        call this method directly, it's generally expected to be used
        internally.

        This is a public method as an extension may need to add/alter
        additional `BlockProcessors` which call this method to recursively
        parse a nested block.

        Arguments:
            parent: The parent element.
            blocks: The blocks of text to parse.

        """
        while blocks:
            for processor in self.blockprocessors:
                if processor.test(parent, blocks[0]):
                    if processor.run(parent, blocks) is not False:
                        break