class NotInWorkingDirectory(PathError):
    _fmt = '"%(path)s" is not in the working directory %(extra)s'