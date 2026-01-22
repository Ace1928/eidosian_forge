import markdown
class SemiSaneOListProcessor(markdown.blockprocessors.OListProcessor):
    SIBLING_TAGS = ['ol']