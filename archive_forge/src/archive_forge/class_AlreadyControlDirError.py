class AlreadyControlDirError(PathError):
    _fmt = 'A control directory already exists: "%(path)s".'