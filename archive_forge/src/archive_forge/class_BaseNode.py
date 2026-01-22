class BaseNode(Node):

    def __init__(self, s):
        super().__init__()
        self.s = str(s)

    def to_string(self, with_parens=None):
        return self.s