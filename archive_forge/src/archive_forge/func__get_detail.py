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