from . import errors
class AlreadyBuilding(errors.BzrError):
    _fmt = 'The tree builder is already building a tree.'