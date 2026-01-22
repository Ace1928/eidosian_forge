import re
import markdown
import xml.etree.ElementTree as etree
class AutolinkExtension(markdown.Extension):
    """
    An extension that turns URLs into links.
    """

    def extendMarkdown(self, md):
        md.inlinePatterns.register(AutolinkPattern(URL_RE, md), 'gfm-autolink', 100)