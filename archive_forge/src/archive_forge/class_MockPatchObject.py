import fixtures
class MockPatchObject(_Base):
    """Deal with code around mock."""

    def __init__(self, obj, attr, new=None, **kwargs):
        super(MockPatchObject, self).__init__()
        if new is None:
            new = mock.DEFAULT
        self._get_p = lambda: mock.patch.object(obj, attr, new, **kwargs)