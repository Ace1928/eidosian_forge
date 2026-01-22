class VersionedFileInvalidChecksum(VersionedFileError):
    _fmt = 'Text did not match its checksum: %(msg)s'