import markdown
class SemiSaneUListProcessor(markdown.blockprocessors.UListProcessor):
    SIBLING_TAGS = ['ul']