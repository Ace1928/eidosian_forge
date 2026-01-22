import re
def _munge_whitespace(self, text):
    """_munge_whitespace(text : string) -> string

        Munge whitespace in text: expand tabs and convert all other
        whitespace characters to spaces.  Eg. " foo\\tbar\\n\\nbaz"
        becomes " foo    bar  baz".
        """
    if self.expand_tabs:
        text = text.expandtabs(self.tabsize)
    if self.replace_whitespace:
        text = text.translate(self.unicode_whitespace_trans)
    return text