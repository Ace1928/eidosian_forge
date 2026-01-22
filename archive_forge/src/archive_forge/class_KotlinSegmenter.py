from typing import TYPE_CHECKING
from langchain_community.document_loaders.parsers.language.tree_sitter_segmenter import (  # noqa: E501
class KotlinSegmenter(TreeSitterSegmenter):
    """Code segmenter for Kotlin."""

    def get_language(self) -> 'Language':
        from tree_sitter_languages import get_language
        return get_language('kotlin')

    def get_chunk_query(self) -> str:
        return CHUNK_QUERY

    def make_line_comment(self, text: str) -> str:
        return f'// {text}'