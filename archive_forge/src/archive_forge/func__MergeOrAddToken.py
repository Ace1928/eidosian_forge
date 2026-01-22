from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import renderer
from prompt_toolkit.token import Token
def _MergeOrAddToken(self, text, token_type):
    """Merges text if the previous token_type matches or appends a new token."""
    if not text:
        return
    if not self._tokens or self._tokens[-1][self.TOKEN_TYPE_INDEX] != token_type:
        self._tokens.append((token_type, text))
    elif self._tokens[-1][self.TOKEN_TYPE_INDEX] == Token.Markdown.Section:
        prv_text = self._tokens[-1][self.TOKEN_TEXT_INDEX]
        prv_indent = re.match('( *)', prv_text).group(1)
        new_indent = re.match('( *)', text).group(1)
        if prv_indent == new_indent:
            self._tokens[-1] = (token_type, text)
        else:
            self._NewLine()
            self._tokens.append((token_type, text))
    else:
        self._tokens[-1] = (token_type, self._tokens[-1][self.TOKEN_TEXT_INDEX] + text)