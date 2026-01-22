from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class toctree(nodes.General, nodes.Element, translatable):
    """Node for inserting a "TOC tree"."""

    def preserve_original_messages(self) -> None:
        rawentries = self.setdefault('rawentries', [])
        for title, _docname in self['entries']:
            if title:
                rawentries.append(title)
        if self.get('caption'):
            self['rawcaption'] = self['caption']

    def apply_translated_message(self, original_message: str, translated_message: str) -> None:
        for i, (title, docname) in enumerate(self['entries']):
            if title == original_message:
                self['entries'][i] = (translated_message, docname)
        if self.get('rawcaption') == original_message:
            self['caption'] = translated_message

    def extract_original_messages(self) -> List[str]:
        messages: List[str] = []
        messages.extend(self.get('rawentries', []))
        if 'rawcaption' in self:
            messages.append(self['rawcaption'])
        return messages