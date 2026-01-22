class RevisionAlreadyPresent(VersionedFileError):
    _fmt = 'Revision {%(revision_id)s} already present in "%(file_id)s".'

    def __init__(self, revision_id, file_id):
        VersionedFileError.__init__(self)
        self.revision_id = revision_id
        self.file_id = file_id