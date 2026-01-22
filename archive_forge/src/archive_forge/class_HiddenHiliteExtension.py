from markdown.extensions.codehilite import CodeHiliteExtension
class HiddenHiliteExtension(CodeHiliteExtension):
    """
    A subclass of CodeHiliteExtension that doesn't highlight on its own.
    """

    def extendMarkdown(self, md, md_globals):
        md.registerExtension(self)