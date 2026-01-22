class InconsistentResolutionOrderError(TypeError):
    """
    The error raised when an invalid IRO is requested in strict mode.
    """

    def __init__(self, c3, base_tree_remaining):
        self.C = c3.leaf
        base_tree = c3.base_tree
        self.base_ros = {base: base_tree[i + 1] for i, base in enumerate(self.C.__bases__)}
        self.base_tree_remaining = base_tree_remaining
        TypeError.__init__(self)

    def __str__(self):
        import pprint
        return '{}: For object {!r}.\nBase ROs:\n{}\nConflict Location:\n{}'.format(self.__class__.__name__, self.C, pprint.pformat(self.base_ros), pprint.pformat(self.base_tree_remaining))