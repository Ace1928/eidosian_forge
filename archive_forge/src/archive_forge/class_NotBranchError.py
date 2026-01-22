class NotBranchError(PathError):
    _fmt = 'Not a branch: "%(path)s"%(detail)s.'

    def __init__(self, path, detail=None, controldir=None):
        from . import urlutils
        path = urlutils.unescape_for_display(path, 'ascii')
        if detail is not None:
            detail = ': ' + detail
        self.detail = detail
        self.controldir = controldir
        PathError.__init__(self, path=path)

    def __repr__(self):
        return '<{} {!r}>'.format(self.__class__.__name__, self.__dict__)

    def _get_format_string(self):
        if self.detail is None:
            self.detail = self._get_detail()
        return super()._get_format_string()

    def _get_detail(self):
        if self.controldir is not None:
            try:
                self.controldir.open_repository()
            except NoRepositoryPresent:
                return ''
            except Exception as e:
                return ': ' + e.__class__.__name__
            else:
                return ': location is a repository'
        return ''