import re
class _SelectorContext:
    parent_map = None

    def __init__(self, root):
        self.root = root