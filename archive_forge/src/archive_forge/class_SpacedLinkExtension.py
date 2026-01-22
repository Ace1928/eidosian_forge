from __future__ import unicode_literals
import markdown
class SpacedLinkExtension(markdown.Extension):
    """
    An extension that supports links and images with additional whitespace.
    """

    def extendMarkdown(self, md, md_globals):
        md.inlinePatterns['link'] = markdown.inlinepatterns.LinkPattern(SPACED_LINK_RE, md)
        md.inlinePatterns['reference'] = markdown.inlinepatterns.ReferencePattern(SPACED_REFERENCE_RE, md)
        md.inlinePatterns['image_link'] = markdown.inlinepatterns.ImagePattern(SPACED_IMAGE_LINK_RE, md)
        md.inlinePatterns['image_reference'] = markdown.inlinepatterns.ImageReferencePattern(SPACED_IMAGE_REFERENCE_RE, md)