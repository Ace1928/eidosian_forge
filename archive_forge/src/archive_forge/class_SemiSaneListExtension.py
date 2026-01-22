import markdown
class SemiSaneListExtension(markdown.Extension):
    """
    An extension that causes lists to be treated the same way GitHub does.
    """

    def extendMarkdown(self, md):
        md.parser.blockprocessors.register(SemiSaneOListProcessor(md.parser), 'olist', 41)
        md.parser.blockprocessors.register(SemiSaneUListProcessor(md.parser), 'ulist', 31)