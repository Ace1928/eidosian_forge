class RootMissing(InternalBzrError):
    _fmt = 'The root entry of a tree must be the first entry supplied to the commit builder.'